#!/usr/bin/env python
# coding: utf-8

# note: in Python, it is recommended to specify the encoding of the file you use. On Python 3 it is automatic.

# this indicates we will use the module xkcd, it is not included in Python, you need to download this library
import xkcd

if __name__ == "__main__":
    print("Opening xkcd joke on web browser...")
    xkcd.Comic(353).show()

    # FYI, you can achieve the same result with the built-in easter egg in python by calling
    # import antigravity (remove comment...)