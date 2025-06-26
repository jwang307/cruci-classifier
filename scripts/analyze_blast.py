#!/usr/bin/env python3
"""
evaluate_blast_hosts.py
————————————————————————————————————————
Aggregate BLASTn TSV results for viral genomes and highlight probable
host associations.

Input  : directory containing files named seqNNNN.tsv
Output : • host_summary.tsv            – one row per (query, host species)
         • top_host_counts.tsv        – count table for host species
         • plots/*.png                – three diagnostic figures
         • console print-outs of key metrics
Usage   : python evaluate_blast_hosts.py <results_dir> [--outdir OUT]
"""

import argparse, glob, re, sys, textwrap
from pathlib import Path
import time
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#### ------------------------------------------------------------------ CLI
def get_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(__doc__))
    p.add_argument("results_dir", help="Folder with seqNNNN.tsv files")
    p.add_argument("--outdir", default="blast_host_report",
                   help="Folder to write summaries & plots (default: %(default)s)")
    p.add_argument("--evalue", type=float, default=0.1,
                   help="Maximum e-value considered significant (default: %(default)s)")
    p.add_argument("--min_pid", type=float, default=30,
                   help="Minimum %%-identity for a hit to be kept (default: %(default)s)")
    return p.parse_args()

#### ------------------------------------------------------------------ helpers
COLS = ["qseqid", "sacc", "staxids", "sscinames",
        "pident", "length", "evalue", "bitscore"]

VIRAL_RE = re.compile(r"(virus|phage|viroid|viri|viral)", re.I)

# Global cache for taxonomy data
TAXONOMY_CACHE = {}

def is_host(name: str) -> bool:
    """heuristic: treat anything *not* containing viral keywords as host"""
    # Ensure name is a string to handle potential NaN values
    return not bool(VIRAL_RE.search(str(name)))

def fetch_taxonomy_lineage(taxid):
    """Fetch taxonomic lineage from NCBI for a given taxid."""
    if pd.isna(taxid):
        return "N/A"
    try:
        # Ensure taxid is an integer for cache key and NCBI query
        taxid_int = int(float(taxid)) # float() handles "123.0"
    except ValueError:
        # If taxid is not a number (e.g. a multi-ID string not handled here, or other text)
        return "Invalid TaxID format"

    if taxid_int in TAXONOMY_CACHE:
        return TAXONOMY_CACHE[taxid_int]

    lineage = "Error fetching lineage" # Default
    try:
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=taxonomy&id={taxid_int}&retmode=xml"
        with urllib.request.urlopen(url) as response:
            if response.getcode() == 200:
                xml_data = response.read()
                root = ET.fromstring(xml_data)
                lineage_element = root.find(".//Lineage")
                if lineage_element is not None and lineage_element.text:
                    lineage = lineage_element.text
                else:
                    # Fallback if Lineage is not found, might indicate an issue with the TaxID or NCBI record
                    sciname_element = root.find(".//ScientificName")
                    if sciname_element is not None and sciname_element.text:
                         lineage = f"Lineage not found for {sciname_element.text} (TaxID: {taxid_int})"
                    else:
                        lineage = f"Lineage and ScientificName not found for TaxID: {taxid_int}"
            else:
                lineage = f"HTTP Error {response.getcode()} for TaxID {taxid_int}"
        
        TAXONOMY_CACHE[taxid_int] = lineage
        time.sleep(0.35)  # Respect NCBI rate limits (aim for <3 requests/sec)
    except urllib.error.HTTPError as e:
        lineage = f"HTTP Error {e.code} for TaxID {taxid_int}: {e.reason}"
        TAXONOMY_CACHE[taxid_int] = lineage 
    except urllib.error.URLError as e:
        lineage = f"URL Error for TaxID {taxid_int}: {e.reason}"
        TAXONOMY_CACHE[taxid_int] = lineage
    except ET.ParseError:
        lineage = f"XML Parse Error for TaxID {taxid_int}"
        TAXONOMY_CACHE[taxid_int] = lineage
    except Exception as e:
        lineage = f"Unexpected error for TaxID {taxid_int}: {str(e)}"
        TAXONOMY_CACHE[taxid_int] = lineage
        
    return lineage

def read_one(path: Path) -> pd.DataFrame:
    """Read one BLAST TSV file"""
    df = pd.read_csv(path, sep="\\t", header=None,
                     names=COLS, dtype={"qseqid": str, "sacc": str})
    df["file_src"] = path.name
    return df

def load_all(dir_: Path) -> pd.DataFrame:
    paths = sorted(Path(dir_).glob("seq*.tsv"))
    if not paths:
        sys.exit("❗ No seq*.tsv files found – check path.")
    dfs = [read_one(p) for p in paths]
    return pd.concat(dfs, ignore_index=True)

#### ------------------------------------------------------------------ main workflow
def main():
    args = get_args()
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    plotdir = outdir / "plots"
    plotdir.mkdir(exist_ok=True)

    df = load_all(args.results_dir)

    # Filter on significance & identity
    df = df.query("evalue <= @args.evalue and pident >= @args.min_pid").copy()
    df["is_host"]  = df["sscinames"].apply(is_host)
    df["is_viral"] = ~df["is_host"]

    # ── keep best hit per query × subject species ──
    df["rank"] = df.groupby(["qseqid", "sscinames"])["bitscore"] \
                   .rank(method="first", ascending=False)
    best = df.query("rank == 1").drop(columns="rank")

    # ------------------------------------------------------------------- summaries
    host_hits = best[best["is_host"]].copy() # Use .copy() to avoid SettingWithCopyWarning
    viral_hits = best[best["is_viral"]]

    # Fetch taxonomy for host hits
    if not host_hits.empty:
        print(f"\nFetching taxonomy for {host_hits['staxids'].nunique()} unique host taxIDs (this may take a while)...")
        # Ensure the new column is created safely, even if host_hits is later empty or staxids are all NaN
        host_hits.loc[:, "lineage"] = host_hits["staxids"].apply(fetch_taxonomy_lineage)
        print("Taxonomy fetching complete.")
    else:
        # Add lineage column even if no host hits, to maintain consistent columns
        host_hits["lineage"] = pd.Series(dtype='object')

    # 1) how many queries have ≥1 host hit?
    queries_with_host = host_hits["qseqid"].nunique()
    total_queries     = best["qseqid"].nunique()

    # 2) host hit counts
    if not host_hits.empty:
        host_counts = (host_hits.groupby(["sscinames", "lineage"])["qseqid"]
                       .nunique()
                       .sort_values(ascending=False)
                       .reset_index(name="query_count"))
    else:
        host_counts = pd.DataFrame(columns=["sscinames", "lineage", "query_count"])

    # 3) export
    host_hits.to_csv(outdir / "host_summary.tsv", sep="\t", index=False)
    host_counts.to_csv(outdir / "top_host_counts.tsv", sep="\t", index=False)

    # ------------------------------------------------------------------- console report
    print("\n=== BLAST host-association summary ===")
    print(f"Total query genomes analysed       : {total_queries}")
    print(f"Queries with ≥1 significant host hit: {queries_with_host} "
          f"({queries_with_host/total_queries:.1%})")
    print("\nTop 10 putative host species:")
    print(host_counts.head(10).to_string(index=False))

    # ------------------------------------------------------------------- plots
    sns.set_style("whitegrid")

    # (a) barplot of top 20 hosts
    plt.figure()
    sns.barplot(data=host_counts.head(20),
                x="query_count", y="sscinames", orient="h")
    plt.xlabel("# viral genomes with hit")
    plt.ylabel("Host species")
    plt.title("Top 20 putative hosts")
    plt.tight_layout()
    plt.savefig(plotdir / "top20_hosts.png", dpi=300)

    # (b) violin / box of %identity
    plt.figure()
    melt = pd.melt(best,
                   id_vars=["pident"],
                   value_vars=["is_host", "is_viral"],
                   var_name="category_flag", value_name="flag_value")
    melt = melt[melt["flag_value"]]
    melt["Category"] = melt["category_flag"].map(
        {"is_host": "Host hits", "is_viral": "Viral hits"})
    sns.violinplot(data=melt, x="Category", y="pident", inner="quartile")
    plt.ylabel("% identity")
    plt.title("% identity distribution")
    plt.tight_layout()
    plt.savefig(plotdir / "pid_violin.png", dpi=300)

    # (c) scatter of length vs pid for host hits
    plt.figure(figsize=(6,4.5))
    sns.scatterplot(data=host_hits,
                    x="length", y="pident",
                    hue="evalue", size="bitscore",
                    palette="viridis_r", alpha=0.7, legend=False)
    plt.xlabel("Alignment length (nt)")
    plt.ylabel("% identity")
    plt.title("Host-hit alignment quality")
    plt.tight_layout()
    plt.savefig(plotdir / "host_len_vs_pid.png", dpi=300)

    print(f"\n✔ Results written to: {outdir.resolve()}")
    print("  ↳ Summary tables  : host_summary.tsv, top_host_counts.tsv")
    print("  ↳ Plots           : plots/*.png\n")

if __name__ == "__main__":
    main()
