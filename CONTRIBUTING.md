# CONTRIBUTING.md

# 1.  Welcome & Scope

`saev` is research code.
PRs that fix bugs, add datasets, or improve docs are welcome.
Large architectural rewrites: please open a discussion first.”

# 2.  TL;DR

Install [uv](https://docs.astral.sh/uv/).
Clone this repository, then from the root directory:

```sh
uv run python -m saev --help
```

You also need [yek](https://github.com/bodo-run/yek) and [lychee](https://github.com/lycheeverse/lychee) for generating docs.

If you want to do any of the web interface work, you need [elm](https://guide.elm-lang.org/install/elm.html), [elm-format](https://github.com/avh4/elm-format/releases/latest) and [tailwindcss](https://github.com/tailwindlabs/tailwindcss/releases/latest).

# 4.  Repo Layout & Data Dependencies

```
src/
  saev/
    __main__.py  <- entrypoint for entire package.
    config.py    <- All configs, which define the CLI flags.

    activations.py  <-
    training.py     <-
    visuals.py      <-

    nn/
      modeling.py    <-
      objectives.py  <-

  docs/
    templates/  <-

  web/
    apps/  <- HTML/CSS and JS (compiled Elm)
    src/   <- Elm source code
```

# 5.  Coding Style & Conventions

* File descriptors from `open()` are called `fd`.
* Use types where possible, including `jaxtyping` hints.
* Decorate functions with `beartype.beartype` unless they use a `jaxtyping` hint, in which case use `jaxtyped(typechecker=beartype.beartype)`.
* Variables referring to a filepath should be suffixed with `_fpath`. Directories are `_dpath`.
* Prefer `make` over `build` when naming functions that construct objects, and use `get` when constructing primitives (like string paths or config values).
* Only use `setup` for naming functions that don't return anything.

Throughout the code, variables are annotated with shape suffixes, as [recommended by Noam Shazeer](https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd).

The key for these suffixes:

* B: batch size
* W: width in patches (typically 14 or 16)
* H: height in patches (typically 14 or 16)
* D: ViT activation dimension (typically 768 or 1024)
* S: SAE latent dimension (768 x 16, etc)
* L: Number of latents being manipulated at once (typically 1-5 at a time)
* C: Number of classes in ADE20K (151)

For example, an activation tensor with shape (batch, width, height d_vit) is `acts_BWHD`.

# 6.  Testing & Linting

# 7.  Commit / PR Checklist

# 8.  Research Reproducibility Notes

# 9.  Asking for Help / Discussion Channels

# 10. Code of Conduct & License Footnotes
