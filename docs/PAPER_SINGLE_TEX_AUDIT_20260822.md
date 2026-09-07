# Single-file LaTeX integration audit

Date: 2026-08-22 (Asia/Shanghai)

## Outcome

- Source: `docs/paper_revision/GNN_IEEE_REVISED_SINGLE.tex`
- Inlined sources: 11 (front matter, figures, body, conclusion, and six tables)
- Remaining `\input{...}` statements: 0
- TeX SHA-256: `0c1e83a0b2b500a296837fe66c81b9779699f8bdead011db2dd9ed1008735a12`
- PDF: `docs/paper_revision/GNN_IEEE_REVISED_SINGLE.pdf`
- PDF size/pages: 277144 bytes / 7 pages
- PDF SHA-256: `eff625cbb972ec1a2359dbcaf136244d247a5c419e50ce719e01ab48ae5f6f70`
- `latexmk`/BibTeX completed successfully; the log contains no undefined
  citation or reference and no fatal LaTeX error.

The merger is reproducible through `evaluation/merge_paper_tex.py`. The
bibliography remains in `ref.bib`, as required by BibTeX; “single TeX” means
that all manuscript and table TeX fragments are inlined, not that the class or
bibliography database is embedded into the TeX source.

## Required assets

- `IEEEtaes.cls`, SHA-256
  `28348eb106afca49f568180602d44611ffc8cb21145f217cb6e8fa0ac99a354c`
- `IEEEtaes.bst`, SHA-256
  `ce2e0493ddd76454e21ebff5a4fa923211332b3a172711b22ea9e8d7f0714b8c`
- `ref.bib`, SHA-256
  `e64c7be69302266988eee53ce4648de413f21d27ed1ac2dc816db19674f88dda`

`IEEEtaes.bst` was absent from `revision_20260819`; the copy bundled in
`GNN_version3/submit-IEEE` was reused so that BibTeX could run.

## Unresolved template/layout issues

1. The supplied file is named `IEEEtaes.cls`, but line 207 declares
   `\ProvidesClass{IEEEphot}`. LaTeX consequently warns that the requested
   class is `IEEEtaes` while the provided class is `IEEEphot`. The log also
   reports IEEE Photonics paper dimensions and an unused `doublecolumn` option.
   The correct journal class must be confirmed before submission.
2. The PDF builds, but the log reports eight overfull boxes. The most severe
   ones are caused by long equations and full SHA-256 strings. These are layout
   tasks after the class is corrected; adjusting them against the wrong class
   would be wasted work.
3. `subcaption` warns that the document class is unknown. This may disappear or
   require a TAES-compatible figure layout after the correct class is supplied.
4. Visual spot checks confirm that page 1 is readable, but page 5 has a visible
   table/text overlap and overflowing hashes. Thus successful compilation is a
   source-integrity check only, not a passed page-layout review.
