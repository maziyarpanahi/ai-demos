# Contributing

Great demos make a powerful idea feel obvious. Contributions should stay
small, runnable, honest, and easy to change.

## The demo contract

Every demo must:

1. Teach one clear technique or produce one clear outcome.
2. Be self-contained inside one `<category>/<outcome>/` directory.
3. Have one obvious entry point.
4. Include copy-paste setup and run commands.
5. Explain the expected output and the main implementation steps.
6. State whether inputs stay local or are sent to a remote endpoint.
7. Contain no secrets, private URLs, logs, receipts, or private user data.
8. Use only repository-owned, licensed, or synthetic assets with clear
   attribution when required.

## Choose a durable home

Use one primary learning goal as the category and an outcome-first directory
name:

```text
vision/draw-object-boxes/
audio/transcribe-speakers/
documents/extract-tables/
agents/research-with-tools/
```

Do not organize the path around a provider. Models and providers change; the
thing a person learns should remain easy to find.

Only add a new top-level category when existing categories genuinely do not
fit. Do not create empty category directories.

## README shape

Keep each demo README short and use this order:

1. **Outcome:** one sentence describing what goes in and what comes out.
2. **Why it is useful:** the transferable ideas the demo teaches.
3. **Requirements:** runtime, endpoint capabilities, and dependencies.
4. **Setup:** copy-paste commands from a fresh clone.
5. **Run:** one minimal command followed by useful options.
6. **Expected output:** the created file or terminal result.
7. **How it works:** three to seven concrete implementation steps.
8. **Make it yours:** the best places to modify the prompt, schema, or UI.
9. **Privacy and limitations:** remote data flow, cost, variability, and
   unsupported claims.

If a social post inspired the public version, explain what was simplified.
Link the original post only when there is a stable public URL.

## Keep dependencies simple

- Prefer standard APIs and direct dependencies.
- Use a separate virtual environment per demo.
- Document the minimum Python or runtime version.
- Record the exact model identifier and dependency versions used for a
  published result when known.
- List verified model names in the demo’s existing **Model(s)** cell. Do not
  duplicate a catalog row just to name another compatible model.
- Add a lock file only when exact transitive versions are important to the
  technique.
- Avoid shared repository packages unless several demos truly cannot remain
  understandable without one.

## Branches and change descriptions

Name branches and changes after the user-visible outcome:

```text
demo/vision-depth-map
fix/object-box-coordinate-clamping
docs/quick-start
```

Pull request titles and descriptions should explain what a reader can now
build, run, or understand. Keep implementation tooling out of public-facing
names and summaries.

## Before opening a pull request

- [ ] Run the demo from its documented setup.
- [ ] Verify the expected output is accurate.
- [ ] Remove secrets, local paths, generated logs, and private data.
- [ ] Check all example assets and links.
- [ ] Add the demo and its verified model name to `CATALOG.md` and the root
      `README.md` without duplicating an existing demo row.
- [ ] Confirm the folder and title describe the outcome.
- [ ] Keep unrelated production assets out of the change.

By contributing code, you agree that it is released under the repository’s
[MIT License](LICENSE).
