"""Filter expressions for job queries

This module provides a filter expression parser for querying jobs by state,
tags, and other attributes.
"""

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from collections.abc import Callable, Iterable
import pyparsing as pp

if TYPE_CHECKING:
    from experimaestro.scheduler.state_provider import MockJob

# Type alias for tags map: job_id -> {tag_key: tag_value}
TagsMap = dict[str, dict[str, str]]


@dataclass
class FilterContext:
    """Context for filter evaluation containing experiment-scoped data

    Attributes:
        tags_map: Maps job identifiers to their tags dict for the current experiment/run
    """

    tags_map: TagsMap = field(default_factory=dict)


# --- classes for processing


def _as_text(value) -> str | None:
    """Text form of a value for the text operators (``=``, ``~``, ``in``)

    Tags keep their type (e.g. ``int``), but queries compare them with quoted
    strings: ``phase = '1'`` must match the tag ``phase=1``.
    """
    return None if value is None else str(value)


class VarExpr:
    def __init__(self, values):
        (self.varname,) = values

    def get(self, job: "MockJob", ctx: FilterContext):
        if self.varname == "@state":
            return job.state.name if job.state else None

        if self.varname == "@name":
            return str(job.path.parent.name)

        # Tags are stored in JobTagModel, accessed via ctx.tags_map keyed by job identifier
        job_tags = ctx.tags_map.get(job.identifier, {})
        return job_tags.get(self.varname, None)

    def __repr__(self):
        return f"""VAR<{self.varname}>"""


class BaseInExpr:
    def __init__(self, values):
        self.var, *stringList = values
        self.values = set(stringList)


class InExpr(BaseInExpr):
    def filter(self, job: "MockJob", ctx: FilterContext):
        return _as_text(self.var.get(job, ctx)) in self.values

    def __repr__(self):
        return f"""IN<{self.var}|{",".join(self.values)}>"""


class NotInExpr(BaseInExpr):
    def filter(self, job: "MockJob", ctx: FilterContext):
        return _as_text(self.var.get(job, ctx)) not in self.values

    def __repr__(self):
        return f"""NOT_IN<{self.var}|{",".join(self.values)}>"""


class RegexExpr:
    def __init__(self, tokens):
        self.var, expr = tokens
        self.regex = re.compile(expr)

    def __repr__(self):
        return f"""REGEX[{self.var}, {self.regex.pattern}]"""

    def filter(self, job: "MockJob", ctx: FilterContext):
        value = _as_text(self.var.get(job, ctx))
        if value is None:
            return False

        return self.regex.match(value) is not None


class ConstantString:
    def __init__(self, tokens):
        (self.value,) = tokens

    def get(self, _job: "MockJob", _ctx: FilterContext):
        return self.value

    def __repr__(self):
        return f'"{self.value}"'


class EqExpr:
    def __init__(self, tokens):
        self.var1, self.var2 = tokens

    def __repr__(self):
        return f"""EQ[{self.var1}, {self.var2}]"""

    def filter(self, job: "MockJob", ctx: FilterContext):
        return _as_text(self.var1.get(job, ctx)) == _as_text(self.var2.get(job, ctx))


class LogicExpr:
    """Logical expression"""

    def __init__(self, tokens):
        self.operator, self.y = tokens
        self.x = None

    def filter(self, job: "MockJob", ctx: FilterContext):
        if self.operator == "and":
            return self.y.filter(job, ctx) and self.x.filter(job, ctx)

        return self.y.filter(job, ctx) or self.x.filter(job, ctx)

    @staticmethod
    def summary(tokens):
        if len(tokens) == 1:
            return tokens[0]
        v = tokens[1]
        v.x = tokens[0]
        for token in tokens[2:]:
            token.x = v
            v = token
        return v

    @staticmethod
    def generator(tokens):
        if len(tokens) == 1:
            return tokens[0]
        v = LogicExpr(("and", tokens[1]))
        v.x = tokens[0]
        return v

    def __repr__(self):
        return f"""{self.x} [{self.operator}] {self.y}"""


# --- Vocabulary (used by the grammar and by the completion)

#: Fields that are not tags, with their description
SPECIAL_FIELDS: dict[str, str] = {
    "@state": "job state",
    "@name": "task identifier",
}

#: Comparison operators (as typed, with the text inserted when completing)
COMPARISON_OPERATORS: dict[str, tuple[str, str]] = {
    "=": ("= ", "equals"),
    "~": ("~ ", "matches a regular expression (anchored at the start)"),
    "in": ("in [", "is in a list"),
    "not in": ("not in [", "is not in a list"),
}

#: Operators combining conditions
BOOLEAN_OPERATORS = ("and", "or")


# --- Grammar

lit = pp.Literal

lpar, rpar, lbra, rbra, eq, comma, pipe, tilde = map(pp.Suppress, "()[]=,|~")
quotedString = pp.QuotedString('"', unquoteResults=True) | pp.QuotedString(
    "'", unquoteResults=True
)

# Tag names can contain letters, digits, underscores, and hyphens
# First character must be a letter, rest can include digits, underscores, hyphens
tag_name = pp.Word(pp.alphas, pp.alphanums + "_-")
var = pp.MatchFirst([lit(name) for name in SPECIAL_FIELDS]) | tag_name
var.setParseAction(VarExpr)

regexExpr = var + tilde + quotedString
regexExpr.setParseAction(RegexExpr)

# A copy is needed: setting the parse action on ``quotedString`` itself would
# turn every quoted string (regexes, ``in`` lists) into a ConstantString
varQuotedString = quotedString.copy()
varQuotedString.setParseAction(ConstantString)
eqExpr = var + eq + (var | varQuotedString)
eqExpr.setParseAction(EqExpr)

stringList = quotedString + pp.ZeroOrMore(comma + quotedString)

notInExpr = var + (pp.Suppress(lit("not in")) + lbra + stringList + rbra)
notInExpr.setParseAction(NotInExpr)

inExpr = var + (pp.Suppress(lit("in")) + lbra + stringList + rbra)
inExpr.setParseAction(InExpr)

matchExpr = eqExpr | regexExpr | inExpr | notInExpr

booleanOp = pp.MatchFirst([lit(op) for op in BOOLEAN_OPERATORS])
logicExpr = (
    matchExpr + pp.ZeroOrMore((booleanOp + matchExpr).setParseAction(LogicExpr))
).setParseAction(LogicExpr.summary)
parenExpr = logicExpr | (lpar + logicExpr + rpar)

filterExpr = (
    parenExpr + pp.ZeroOrMore((booleanOp + parenExpr).setParseAction(LogicExpr))
).setParseAction(LogicExpr.summary)
expr = (matchExpr + pp.Optional(pipe + filterExpr)).setParseAction(LogicExpr.generator)


def createFilter(query: str, ctx: FilterContext = None) -> Callable[["MockJob"], bool]:
    """Returns a filter function given a query string

    Args:
        query: Filter expression (e.g., '@state = "done" and model = "bm25"')
        ctx: FilterContext containing tags map and other experiment-scoped data.
             If None, an empty context is used.

    Returns:
        A callable that takes a MockJob and returns True if it matches.
    """
    if ctx is None:
        ctx = FilterContext()
    (r,) = logicExpr.parseString(query, parseAll=True)

    def filter_fn(job: "MockJob") -> bool:
        return r.filter(job, ctx)

    return filter_fn


#: Short syntax reminder shown by the TUI and the web UI next to the search box
FILTER_SYNTAX_HINT = (
    "Syntax: @state = 'done' | @name ~ 'regex' | tag = 'value' | and/or"
)


def filter_jobs(
    query: str, jobs: Iterable["MockJob"], tags_map: TagsMap | None = None
) -> list["MockJob"]:
    """Return the jobs matching a filter expression

    Args:
        query: Filter expression (see :func:`createFilter`)
        jobs: Jobs to filter
        tags_map: Tags of the jobs (job identifier -> tags) for the
            experiment/run the jobs belong to

    Raises:
        pyparsing.ParseException: if the query is not a valid expression
    """
    filter_fn = createFilter(query, FilterContext(tags_map=tags_map or {}))
    return [job for job in jobs if filter_fn(job)]


# --- Completion


def state_names() -> list[str]:
    """Values of ``@state``, in lifecycle order"""
    from experimaestro.scheduler.interfaces import JobState

    return [
        state.name
        for state in (
            JobState.UNSCHEDULED,
            JobState.WAITING,
            JobState.READY,
            JobState.SCHEDULED,
            JobState.RUNNING,
            JobState.DONE,
            JobState.ERROR,
        )
    ]


#: Tag keys that can be written in an expression (see ``tag_name``)
_TAG_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*\Z")


@dataclass
class FilterVocabulary:
    """Values suggested when completing an expression

    Attributes:
        tags: Values of each tag key
        names: Task identifiers (values of ``@name``)
        states: Values of ``@state``
    """

    tags: dict[str, set[str]] = field(default_factory=dict)
    names: set[str] = field(default_factory=set)
    states: list[str] = field(default_factory=state_names)

    def add_jobs(self, jobs: Iterable["MockJob"], tags_map: TagsMap) -> None:
        """Adds the task identifiers and tags of jobs (tags from ``tags_map``)"""
        for job in jobs:
            self.names.add(str(job.path.parent.name))
            for key, value in tags_map.get(job.identifier, {}).items():
                self.tags.setdefault(key, set()).add(str(value))

    def values(self, field_name: str) -> list[str]:
        """Possible values of a field"""
        match field_name:
            case "@state":
                return list(self.states)
            case "@name":
                return sorted(self.names)
        return sorted(self.tags.get(field_name, ()))


@dataclass
class FilterSuggestion:
    """A completion: ``text`` replaces the range of the completion"""

    label: str
    text: str
    kind: str  # field, operator, value or keyword
    detail: str = ""


@dataclass
class FilterCompletion:
    """Completions of the text between ``start`` and ``end`` of a query"""

    start: int
    end: int
    items: list[FilterSuggestion]

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "items": [
                {
                    "label": item.label,
                    "text": item.text,
                    "kind": item.kind,
                    "detail": item.detail,
                }
                for item in self.items
            ],
        }


_COMPLETION_TOKEN = re.compile(
    r"""(?P<ws>\s+)
    |(?P<string>'[^']*'|"[^"]*")
    |(?P<open_string>['"].*)
    |(?P<word>@?[A-Za-z][A-Za-z0-9_-]*|@)
    |(?P<punct>.)""",
    re.VERBOSE | re.DOTALL,
)

#: Maximum number of suggested values
MAX_SUGGESTIONS = 50


def _quote(value: str, quote: str = "'") -> str | None:
    """Quotes a value (the grammar has no escapes: None if impossible)"""
    if quote in value:
        quote = '"' if quote == "'" else "'"
        if quote in value:
            return None
    return f"{quote}{value}{quote}"


def _matching(candidates: Iterable[str], partial: str) -> list[str]:
    """Candidates starting with ``partial``, then those containing it (from two
    characters on), ignoring case"""
    partial = partial.lower()
    prefix, inner = [], []
    for candidate in candidates:
        lowered = candidate.lower()
        if lowered.startswith(partial):
            prefix.append(candidate)
        elif len(partial) > 1 and partial in lowered:
            inner.append(candidate)
    return prefix + inner


def _expected(tokens: list[tuple[str, str]]) -> tuple[str, str | None]:
    """What comes after complete tokens: the state of the parser, and the
    field being compared"""
    state, current = "field", None
    for kind, value in tokens:
        if kind == "ws":
            continue
        match state, kind, value:
            case "field", "word", _ if (
                not value.startswith("@") or value in SPECIAL_FIELDS
            ):
                state, current = "operator", value
            case "operator", "punct", "=":
                state = "value"
            case "operator", "punct", "~":
                state = "regex"
            case "operator", "word", "in":
                state = "lbra"
            case "operator", "word", "not":
                state = "in"
            case "in", "word", "in":
                state = "lbra"
            case "lbra", "punct", "[":
                state = "list_value"
            case "list_value", "string", _:
                state = "list_next"
            case "list_next", "punct", ",":
                state = "list_value"
            case "list_next", "punct", "]":
                state = "boolean"
            case "value" | "regex", "string", _:
                state = "boolean"
            case "value", "word", _:
                # Comparison of two fields
                state = "boolean"
            case "boolean", "word", _ if value in BOOLEAN_OPERATORS:
                state, current = "field", None
            case _:
                return "invalid", None
    return state, current


def complete_filter(
    query: str, cursor: int | None = None, vocabulary: FilterVocabulary | None = None
) -> FilterCompletion:
    """Completions of a filter expression at a cursor position

    The suggestions depend on what the expression expects at the cursor: a
    field (``@state``, ``@name`` or a tag key), a comparison operator, a value
    of the field (states, task identifiers or tag values), or a boolean
    operator.

    Args:
        query: The expression being typed
        cursor: Position of the cursor (end of the query by default)
        vocabulary: Tag keys and values, task identifiers to suggest
    """
    if vocabulary is None:
        vocabulary = FilterVocabulary()
    cursor = len(query) if cursor is None else max(0, min(cursor, len(query)))

    tokens = [
        (m.lastgroup, m.group()) for m in _COMPLETION_TOKEN.finditer(query[:cursor])
    ]
    start = end = cursor
    partial_kind, partial = None, ""
    if tokens and tokens[-1][0] in ("word", "open_string"):
        partial_kind, partial = tokens.pop()
        start = cursor - len(partial)
        rest = query[cursor:]
        if partial_kind == "word":
            end += len(re.match(r"[A-Za-z0-9_-]*", rest).group())
        elif (closing := rest.find(partial[0])) >= 0:
            end += closing + 1

    state, current = _expected(tokens)
    # Separates an operator from the previous token
    space = "" if start == 0 or query[start - 1].isspace() else " "

    items: list[FilterSuggestion] = []
    match state:
        case "value" | "regex" | "list_value":
            # An unquoted word is completed too
            if partial_kind == "open_string":
                quote, partial = partial[0], partial[1:]
            else:
                quote = "'"
            items = _value_items(vocabulary, current, state, partial, quote)
        case _ if partial_kind == "open_string":
            pass
        case "field":
            fields = dict(SPECIAL_FIELDS)
            for key in sorted(vocabulary.tags):
                if key not in fields and _TAG_NAME_RE.match(key):
                    fields[key] = "tag"
            items = [
                FilterSuggestion(name, f"{name} ", "field", fields[name])
                for name in _matching(fields, partial)
            ]
        case "operator":
            items = [
                FilterSuggestion(op, f"{space}{text}", "operator", detail)
                for op, (text, detail) in COMPARISON_OPERATORS.items()
                if op.startswith(partial)
            ]
        case "in" if "in".startswith(partial):
            items = [FilterSuggestion("in", f"{space}in [", "operator", "")]
        case "lbra" if not partial:
            items = [FilterSuggestion("[", "[", "operator", "start of the list")]
        case "list_next" if not partial:
            items = [
                FilterSuggestion(",", ", ", "operator", "another value"),
                FilterSuggestion("]", "] ", "operator", "end of the list"),
            ]
        case "boolean":
            items = [
                FilterSuggestion(op, f"{space}{op} ", "keyword")
                for op in BOOLEAN_OPERATORS
                if op.startswith(partial)
            ]

    return FilterCompletion(start, end, items)


def _value_items(
    vocabulary: FilterVocabulary,
    field_name: str | None,
    state: str,
    partial: str,
    quote: str,
) -> list[FilterSuggestion]:
    """Suggested (quoted) values of a field"""
    if field_name is None:
        return []
    values = vocabulary.values(field_name)
    if state == "regex":
        values = [re.escape(value) for value in values]
    detail = SPECIAL_FIELDS.get(field_name, "tag value")
    # A space after a single value, for the boolean operator that follows
    suffix = "" if state == "list_value" else " "
    items = []
    for value in _matching(values, partial)[:MAX_SUGGESTIONS]:
        if (quoted := _quote(value, quote)) is not None:
            items.append(FilterSuggestion(quoted, f"{quoted}{suffix}", "value", detail))
    return items
