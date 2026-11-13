# Graph explorer

`index.html` is a single-file, offline viewer for release 2026-10-13: Explore (search a company, browse its
suppliers, customers and offerings), Analyze, and Accuracy tabs. The graph is embedded in the page; the only external
dependency is D3 from cdnjs.

Open it directly in a browser, or enable GitHub Pages for this repo and visit `/site/`.

Each edge lists its source documents (SEC filing or a pinned Wikipedia revision) and the extracted claim. The
evidence sentences themselves are not embedded, matching the release, which excludes quotation text; follow the
document link to read the source.
