# Vendored explorer libraries

`report.py` inlines these two minified libraries into every generated `index.html`, so the report's
§1 crossfilter explorer works **fully offline** — the file is a self-contained artifact with no
network dependency at view time (per the disc #4 "renders offline with nothing installed" promise and
the internal reference reports). They are the explorer's only JS dependency; the article and the inline-SVG
figures need no JS at all.

| file | library | version | license | source |
|---|---|---|---|---|
| `d3.min.js` | D3 | 7.8.5 | ISC | https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js |
| `crossfilter.min.js` | Crossfilter | 1.3.12 | Apache-2.0 | https://cdnjs.cloudflare.com/ajax/libs/crossfilter/1.3.12/crossfilter.min.js |

Both are the exact bytes published at those URLs, verified against the SHA-384 Subresource Integrity
hashes the report previously pinned:

- d3.min.js — `sha384-su5kReKyYlIFrI62mbQRKXHzFobMa7BHp1cK6julLPbnYcCW9NIZKJiTODjLPeDh`
- crossfilter.min.js — `sha384-qIMAO7hiswCdbLggCimcQKL2VYr4shEpIgnKP+9GL4YQPHJH4tQ/lLHsaLtDFulA`

To update: re-fetch the pinned version, re-verify the SHA-384, and bump the version here.
