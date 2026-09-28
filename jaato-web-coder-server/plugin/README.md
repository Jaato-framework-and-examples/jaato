# jaato-web-coder-toolchain-offer

The web coder's `toolchain_offer` enrichment plugin (#1344). When a `cli`,
`interactive_shell` or `notebook` result shows a command was not found, and
the web coder offers a toolchain that provides it, the plugin:

- appends one line to the result telling the model which toolchain provides
  the command and that the user binds it from the web coder (or, if it is
  already bound, that something else is wrong);
- sends the page a `tool.result_enriched` notice (`kind: "toolchain_offer"`,
  daemon protocol 1.31), which becomes the Bind chip in the rail.

It reads `<workspace>/.jaato/toolchain-offer.json`, written by
`jaato-web-coder-server` when its `environment:` block is on. A workspace
without that file gets nothing.

Installation: see [INSTALL.md](INSTALL.md).
