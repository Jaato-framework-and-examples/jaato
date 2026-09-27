/**
 * The syntax highlighter the viewer shares: ``lowlight`` (highlight.js
 * producing a syntax tree, not an HTML string) with a fixed set of
 * grammars.  Used by the markdown view's fenced code
 * (``rehypeFenceHighlight``) and by the Files viewer's ``CodeView``; both
 * are lazily loaded, so this module is a chunk of its own that neither the
 * main bundle nor a session that never views a file pays for.
 *
 * Only the grammars below are registered -- lowlight's ``common`` pack
 * would add ~20 more to every viewer load.  Nothing is auto-detected: a
 * language is named (a fence's ``language-*``) or read off the path
 * (``languageForPath`` in ``protocol/codeLanguages``), and anything else stays plain.
 */
import { createLowlight } from "lowlight";
import bash from "highlight.js/lib/languages/bash";
import css from "highlight.js/lib/languages/css";
import diff from "highlight.js/lib/languages/diff";
import dockerfile from "highlight.js/lib/languages/dockerfile";
import go from "highlight.js/lib/languages/go";
import ini from "highlight.js/lib/languages/ini";
import java from "highlight.js/lib/languages/java";
import javascript from "highlight.js/lib/languages/javascript";
import json from "highlight.js/lib/languages/json";
import markdown from "highlight.js/lib/languages/markdown";
import python from "highlight.js/lib/languages/python";
import rust from "highlight.js/lib/languages/rust";
import shell from "highlight.js/lib/languages/shell";
import sql from "highlight.js/lib/languages/sql";
import typescript from "highlight.js/lib/languages/typescript";
import xml from "highlight.js/lib/languages/xml";
import yaml from "highlight.js/lib/languages/yaml";

export const lowlight = createLowlight({ bash, css, diff, dockerfile, go, ini, java, javascript, json, markdown, python, rust, shell, sql, typescript, xml, yaml });
