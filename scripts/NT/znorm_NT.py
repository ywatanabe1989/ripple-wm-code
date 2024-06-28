#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-06-28 16:43:56 (ywatanabe)"

import mngs

CONFIG = mngs.io.load_configs()


def main():

    LPATHS_NT = mngs.gen.natglob(f"./data/Sub_0?/Session_0?/NT/*.npy")
    mngs.gen.natglob(CONFIG["PATH_NT"])

    by = ["by_trial", "by_session"]

    for lpath_nt in LPATHS_NT:
        if not "_z_by" in lpath_nt:
            for _by in by:
                NT = mngs.io.load(lpath_nt)
                dim = 0 if by == "by_session" else -1
                NT_z = mngs.gen.to_z(NT, dim=dim)
                lpath_z = lpath_nt.replace(".npy", f"_z_{_by}.npy")
                mngs.io.save(NT_z, lpath_z, from_cwd=True)


if __name__ == "__main__":
    main()
