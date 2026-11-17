#!/usr/bin/env python3
"""Audit a pinned requirements lockfile for staleness, abandonment, and CVEs.

Stdlib only (plus optional `gh` or $GITHUB_TOKEN/$GH_TOKEN for repo health).
It answers three questions about every pinned package:

  outdated    the pin is not the newest stable release on PyPI
  stale       the newest stable release is old, i.e. upstream may be dormant,
              and for a transitive pin it also means nothing in the graph needs
              anything newer -- the two signals together are the abandonment
              signal (an old-but-active project like `h11` is "done", a
              package nobody has shipped in years *and* whose repo is idle is
              a maintenance liability)
  vulnerable  OSV reports a known advisory for the pinned version

Repo health (archived / last push / open issues) is checked for the direct
requirements and every stale package by default, because that is the set where
an answer changes a decision. `--repos all` covers the whole graph.

Exit status: 0 clean, 1 findings at/above --fail-on, 2 usage/error.

    python scripts/audit_deps.py                       # audit requirements.txt
    python scripts/audit_deps.py -r requirements-dev.txt --repos all
    python scripts/audit_deps.py --stale 540 --fail-on vulnerable
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import re
import subprocess
import sys
import urllib.request
from shutil import which

PYPI = "https://pypi.org/pypi/{name}/json"
OSV = "https://api.osv.dev/v1/query"
# Version suffixes that are not a "real" release for a staleness verdict, e.g.
# 0.8.0rc2 (defusedxml) or 1.0.dev6 (httpx). CalVer (2026.11.15) has no suffix.
PRERELEASE = re.compile(r"(?:\.dev|\.post\.dev|a|b|c|rc|alpha|beta|dev)\d*$", re.I)
GITHUB_URL = re.compile(r"https?://github\.com/([^/]+)/([^/#?\s]+)")
SPONSOR = re.compile(r"github\.com/(sponsors|features/sponsors)(/|$)", re.I)


def parse_lock(path: str) -> tuple[list[tuple[str, str]], set[str]]:
    """Return ([(name, pinned_version)], direct_requirement_names).

    "Direct" means the lockfile attributes the package to the `-r` input file,
    and the name is also listed (unpinned or floored) in a sibling `.in` file.
    """
    pinned: list[tuple[str, str]] = []
    from_input: set[str] = set()
    text = open(path, encoding="utf-8").read()
    for block in re.split(r"\n(?=[A-Za-z0-9._-]+==)", text):
        m = re.match(r"([A-Za-z0-9._-]+)==([^\s\\;]+)", block)
        if not m:
            continue
        pinned.append((m.group(1), m.group(2)))
        if re.search(r"#\s+via[^\n]*-r\s", block):
            from_input.add(m.group(1))
    stem = re.sub(r"\.txt$", ".in", path)
    listed = set()
    if os.path.exists(stem):
        for line in open(stem, encoding="utf-8"):
            line = line.split("#")[0].strip()
            m = re.match(r"([A-Za-z0-9._-]+)\s*[<>=!~]", line) or re.match(
                r"([A-Za-z0-9._-]+)$", line
            )
            if m:
                listed.add(m.group(1).lower())
    direct = {n for n, _ in pinned if n.lower() in listed or n in from_input}
    return pinned, direct


def get_json(url: str, payload: bytes | None = None, token: str = "") -> object:
    headers = {"Content-Type": "application/json", "User-Agent": "mcp-server-dep-audit"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=payload, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def newest_stable(releases: dict, declared: str) -> tuple[str, str]:
    """(version, ISO date) of the newest non-yanked, non-prerelease release."""
    def date_for(version: str) -> str:
        return max((f.get("upload_time_iso_8601") or "" for f in releases.get(version, [])), default="")

    for version in sorted(releases, key=date_for, reverse=True):
        if not releases[version] or PRERELEASE.search(version):
            continue
        if all(f.get("yanked") for f in releases[version]):
            continue
        return version, date_for(version)
    # Every release yanked or pre-release: fall back to what PyPI declares.
    return declared, date_for(declared)


def pypi_report(name: str, pinned: str) -> dict:
    data = get_json(PYPI.format(name=name))
    info, releases = data["info"], data.get("releases", {})
    latest, published = newest_stable(releases, info.get("version", ""))
    pin_files = releases.get(pinned, [])
    urls = repo_candidates(info)
    classifiers = info.get("classifiers", [])
    return {
        "name": name,
        "pinned": pinned,
        "latest_stable": latest or info.get("version", ""),
        "published": published[:10],
        "age_days": (
            (dt.datetime.now(dt.timezone.utc) - dt.datetime.strptime(published[:10], "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)).days
            if published
            else None
        ),
        "pinned_yanked": bool(pin_files) and all(f.get("yanked") for f in pin_files),
        "deprecated": [c for c in classifiers if re.search(r"Archive|Deprecated|Moved", c)],
        "repo": urls[0] if urls else None,
    }


REPO_LABEL_PRIORITY = ("source", "repository", "code", "homepage", "bug tracker", "issues")


def repo_candidates(info: dict) -> list[str]:
    """The package's GitHub URLs, best-labelled first (PyPI's own order is arbitrary).

    Sponsorship links are dropped: `github.com/sponsors/<someone>` is a funding
    page, not the source repository, and looking it up reports a bogus 404.
    """
    labelled = [(str(key), str(url or "")) for key, url in (info.get("project_urls") or {}).items()]
    if info.get("home_page"):
        labelled.append(("homepage", str(info["home_page"])))

    def rank(item: tuple[str, str]) -> tuple[int, int]:
        key = item[0].lower()
        return (next((i for i, p in enumerate(REPO_LABEL_PRIORITY) if p in key), len(REPO_LABEL_PRIORITY)), len(key))

    seen, ordered = set(), []
    for _, url in sorted(labelled, key=rank):
        if not GITHUB_URL.match(url) or SPONSOR.search(url):
            continue
        slug = "/".join(GITHUB_URL.match(url).groups()).removesuffix(".git").lower()
        if slug not in seen:
            seen.add(slug)
            ordered.append(url)
    return ordered


def repo_health(url: str) -> dict:
    m = GITHUB_URL.match(url)
    if not m:
        return {}
    owner, repo = m.group(1), m.group(2).removesuffix(".git")
    path = f"repos/{owner}/{repo}"
    try:
        if which("gh"):
            out = subprocess.run(
                ["gh", "api", path, "--jq", "{archived,pushed_at,stars:.stargazers_count,issues:.open_issues_count}"],
                capture_output=True, text=True, timeout=60,
            )
            if out.returncode == 0:
                data = json.loads(out.stdout)
                data["full_name"] = f"{owner}/{repo}"
                return data
        data = get_json(f"https://api.github.com/{path}", token=github_token())
        data["full_name"] = f"{owner}/{repo}"
        return data
    except Exception as exc:  # noqa: BLE001 - reporting must never crash on metadata
        return {"error": str(exc)[:100]}


def github_token() -> str:
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""


def osv_report(name: str, version: str) -> tuple[list[str], str]:
    """(advisory summaries, query error) for the pinned version.

    A failed lookup is reported separately from an advisory: it must never be
    mistaken for a clean result *or* for a vulnerability.
    """
    body = json.dumps({"package": {"name": name, "ecosystem": "PyPI"}, "version": version}).encode()
    try:
        data = get_json(OSV, payload=body)
    except Exception as exc:  # noqa: BLE001
        return [], f"OSV query failed: {exc}"
    ids = []
    for vuln in data.get("vulns") or []:
        fixed = sorted(
            {
                event.get("fixed", "?")
                for affected in vuln.get("affected", [])
                for rng in affected.get("ranges", [])
                for event in rng.get("events", [])
                if "fixed" in event
            }
        )
        aliases = "/".join(vuln.get("aliases") or []) or vuln.get("id", "?")
        ids.append(f"{aliases} (fixed in {', '.join(fixed) or 'unreleased'})")
    return ids, ""


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-r", "--requirements", default="requirements.txt", help="pinned lockfile to audit")
    parser.add_argument("--stale", type=int, default=365, help="days since the newest stable release that counts as stale (default 365)")
    parser.add_argument("--repos", choices=["auto", "all", "off"], default="auto", help="check upstream repo health for direct+stale packages, everything, or nothing")
    parser.add_argument("--no-osv", action="store_true", help="skip the OSV advisory lookup")
    parser.add_argument("--fail-on", choices=["none", "outdated", "stale", "vulnerable"], default="stale", help="lowest finding severity that sets a non-zero exit status")
    parser.add_argument("--json", action="store_true", help="emit machine-readable findings instead of a table")
    args = parser.parse_args(argv)

    if not os.path.exists(args.requirements):
        print(f"no such lockfile: {args.requirements}", file=sys.stderr)
        return 2
    pinned, direct = parse_lock(args.requirements)
    with cf.ThreadPoolExecutor(16) as pool:
        reports = list(pool.map(lambda nv: pypi_report(*nv), pinned))

    def wants_repo(rep: dict) -> bool:
        if args.repos == "off" or not rep.get("repo"):
            return False
        return args.repos == "all" or rep["name"] in direct or (rep["age_days"] or 0) >= args.stale

    targets = [r for r in reports if wants_repo(r)]
    with cf.ThreadPoolExecutor(8) as pool:
        for rep, health in zip(targets, pool.map(lambda r: repo_health(r["repo"]), targets)):
            rep.update(health or {})

    if not args.no_osv:
        with cf.ThreadPoolExecutor(12) as pool:
            for rep, (vulns, osv_error) in zip(reports, pool.map(lambda r: osv_report(r["name"], r["pinned"]), reports)):
                rep["vulns"] = vulns
                if osv_error:
                    rep["error"] = rep.get("error") or osv_error

    for rep in reports:
        flags = []
        if rep.get("pinned_yanked"):
            flags.append("YANKED-PIN")
        if rep.get("deprecated"):
            flags.append("DEPRECATED-CLASSIFIER")
        if rep.get("archived"):
            flags.append("REPO-ARCHIVED")
        if rep.get("vulns"):
            flags.append("VULNERABLE")
        if (rep["age_days"] or 0) >= args.stale:
            flags.append("STALE")
        elif rep.get("pushed_at") and not rep.get("vulns"):
            pushed = dt.datetime.strptime(rep["pushed_at"][:10], "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
            if (dt.datetime.now(dt.timezone.utc) - pushed).days >= args.stale:
                flags.append("REPO-IDLE")
        if rep["pinned"] != rep["latest_stable"]:
            flags.append("OUTDATED")
        rep["flags"] = flags

    order = {"VULNERABLE": 0, "REPO-ARCHIVED": 1, "DEPRECATED-CLASSIFIER": 2, "YANKED-PIN": 3, "STALE": 4, "REPO-IDLE": 5, "OUTDATED": 6}
    reports.sort(key=lambda r: (min((order.get(f, 9) for f in r["flags"]), default=9), r["name"]))

    if args.json:
        print(json.dumps(reports, indent=1))
    else:
        print(f"# {args.requirements}: {len(reports)} pinned packages ({len(direct)} direct)\n")
        header = f"{'package':26} {'pinned':13} {'latest':13} {'rel_age':>7} {'push':10} {'stars':>6}  flags"
        print(header)
        print("-" * len(header))
        for rep in reports:
            pushed = (rep.get("pushed_at") or "-")[:10]
            stars = rep.get("stars")
            print(
                f"{rep['name']:26} {rep['pinned']:13} {rep['latest_stable']:13} "
                f"{str(rep['age_days']):>7} {pushed:10} {str(stars) if stars is not None else '-':>6}  "
                f"{','.join(rep['flags']) or 'ok'}"
            )
            for vuln in rep.get("vulns") or []:
                print(f"    ! {vuln}")
            if rep.get("error"):
                print(f"    ? repo lookup failed: {rep['error']}")
        serious = [r for r in reports if any(f != "OUTDATED" for f in r["flags"])]
        print(
            f"\n{sum(bool(r['flags']) for r in reports)} flagged, "
            f"{len([r for r in reports if r['flags'] == ['OUTDATED']])} outdated-only, "
            f"{len(serious)} needing an upstream-health decision."
        )
        if serious:
            print("Watch-list: " + ", ".join(sorted(r["name"] for r in serious)))

    # `--fail-on none` never fails; otherwise fail on any flag at or above the
    # chosen severity (a clean package scores 99, above every threshold).
    thresholds = {"none": -1, "outdated": 6, "stale": 4, "vulnerable": 0}
    worst = min(
        (min((order.get(f, 9) for f in r["flags"]), default=99) for r in reports),
        default=99,
    )
    return 1 if worst <= thresholds[args.fail_on] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
