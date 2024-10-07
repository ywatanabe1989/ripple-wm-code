#!./.env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-07-08 20:43:38 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/utils/parse_lpath.py

import re


def parse_lpath(lpath):
    sub = re.search(r"Sub_(\d+)", lpath)
    session = re.search(r"Session_(\d+)", lpath)
    # roi = re.search(r"/([^/]+)\.pkl$", lpath)
    roi = None
    for _roi in ["AHL", "AHR", "PHL", "PHR", "ECL", "ECR", "AL", "AR"]:
        if _roi in lpath:
            roi = _roi

    sub = sub.group(1) if sub else None
    session = session.group(1) if session else None
    # roi = roi.group(1) if roi else None

    return {"sub": sub, "session": session, "roi": roi}


# EOF
