# diffmrc documentation sources

Two LaTeX documents share one style file:

    common/diffmrc-manual.sty   shared style (xelatex, English)
    tutorial/tutorial.tex       the Tutorial (article): first use
    manual/manual.tex           the Manual (report): reference, methods,
                                verification records, development

## Building

Both documents are compiled with xelatex, run twice so that the table
of contents, cross-references and page numbers settle; a document with
many cross-references may need a third pass.

    cd tutorial && xelatex tutorial.tex && xelatex tutorial.tex
    cd ../manual && xelatex manual.tex && xelatex manual.tex

Check the log for `Rerun` prompts after the final pass; a clean pair
of runs ends without them.

## Conventions

- English throughout; formal register, no emphasis symbols.
- Code follows the runnable snippets in the repository scripts, with
  imports shown in full so that a reader can paste a block as is.
- Numbers quoted in the documents must be traceable to a result file
  or a test assertion; when a protocol or basis changes the number,
  the note travels with the number.
