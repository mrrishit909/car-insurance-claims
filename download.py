"""Step 1: freMTPL2freq, the risk features and claim counts of French motor third-party liability policies, from OpenML.

    ./venv/bin/python download.py   -> data/freMTPL2freq.arff, results/pull.json

OpenML dataset 41214 (https://www.openml.org/d/41214), file 20649148. OpenML lists the licence as CC0. The data comes from the CASdatasets R package (Dutang and Charpentier, "CASdatasets", github.com/dutangc/CASdatasets, a GPL-licensed package)
that goes with "Computational Actuarial Science with R" (Charpentier, ed., CRC, 2018). One row per policy: IDpol, ClaimNb (number of claims during the exposure), Exposure (the part of a year the policy was in force), Area, VehPower, VehAge, DrivAge,
BonusMalus, VehBrand, VehGas, Density (inhabitants per square kilometre where the driver lives), Region. About 36 MB, so it is not committed; this script fetches it. (OpenML's description says 677,991 policies; the file has 678,013, the count of the original.)
"""
import hashlib
import json
import re
import ssl
import urllib.request
from datetime import date
from pathlib import Path

import certifi

HERE = Path(__file__).parent
URL = "https://openml.org/data/v1/download/20649148/freMTPL2freq.arff"
API = "https://www.openml.org/api/v1/json/data/41214"
CTX = ssl.create_default_context(cafile=certifi.where())


def main():
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "results").mkdir(exist_ok=True)
    blob = urllib.request.urlopen(urllib.request.Request(URL, headers={"User-Agent": "research-portfolio (Rishit Raj Mathur)"}), context=CTX, timeout=600).read()
    (HERE / "data" / "freMTPL2freq.arff").write_bytes(blob)
    lines = blob.decode("utf-8").splitlines(); start = lines.index("@data") + 1; info = {"pulled": date.today().isoformat(), "url": URL, "bytes": len(blob), "rows": len(lines) - start, "sha256": hashlib.sha256(blob).hexdigest()}
    desc = json.loads(urllib.request.urlopen(urllib.request.Request(API, headers={"User-Agent": "research-portfolio (Rishit Raj Mathur)"}), context=CTX, timeout=120).read())["data_set_description"]
    info["openml_licence"] = desc["licence"]; info["openml_description_policies"] = int(re.search(r"([0-9,]+) motor third-part", desc["description"]).group(1).replace(",", ""))
    (HERE / "results" / "pull.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
