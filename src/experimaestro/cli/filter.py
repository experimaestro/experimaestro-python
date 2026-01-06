"""Filter expressions for job queries

This module provides a filter expression parser for querying jobs by state,
tags, and other attributes.
"""

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, TYPE_CHECKING
import pyparsing as pp

if TYPE_CHECKING:
    from experimaestro.scheduler.state_provider import MockJob

# Type alias for tags map: job_id -> {tag_key: tag_value}
TagsMap = Dict[str, Dict[str, str]]


@dataclass
class FilterContext:
    """Context for filter evaluation containing experiment-scoped data

    Attributes:
        tags_map: Maps job identifiers to their tags dict for the current experiment/run
    """

    tags_map: TagsMap = field(default_factory=dict)


# --- classes for processing


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
        value = self.var.get(job, ctx)
        return value in self.values

    def __repr__(self):
        return f"""IN<{self.varname}|{",".join(self.values)}>"""


class NotInExpr(BaseInExpr):
    def filter(self, job: "MockJob", ctx: FilterContext):
        value = self.var.get(job, ctx)
        return value not in self.values

    def __repr__(self):
        return f"""NOT_IN<{self.varname}|{",".join(self.values)}>"""


class RegexExpr:
    def __init__(self, tokens):
        self.var, expr = tokens
        self.regex = re.compile(expr)

    def __repr__(self):
        return f"""REGEX[{self.varname}, {self.value}]"""

    def matches(self, _manager, publication):
        if self.varname == "tag":
            return self.value in publication.tags

        raise AssertionError()

    def filter(self, job: "MockJob", ctx: FilterContext):
        value = self.var.get(job, ctx)
        if not value:
            return False

        return self.regex.match(value)


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
        return self.var1.get(job, ctx) == self.var2.get(job, ctx)


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


# --- Grammar

lit = pp.Literal

lpar, rpar, lbra, rbra, eq, comma, pipe, tilde = map(pp.Suppress, "()[]=,|~")
quotedString = pp.QuotedString('"', unquoteResults=True) | pp.QuotedString(
    "'", unquoteResults=True
)

# Tag names can contain letters, digits, underscores, and hyphens
# First character must be a letter, rest can include digits, underscores, hyphens
tag_name = pp.Word(pp.alphas, pp.alphanums + "_-")
var = lit("@state") | lit("@name") | tag_name
var.setParseAction(VarExpr)

regexExpr = var + tilde + quotedString
regexExpr.setParseAction(RegexExpr)

varQuotedString = quotedString
varQuotedString.setParseAction(ConstantString)
eqExpr = var + eq + (var | quotedString)
eqExpr.setParseAction(EqExpr)

stringList = quotedString + pp.ZeroOrMore(comma + quotedString)

notInExpr = var + (pp.Suppress(lit("not in")) + lbra + stringList + rbra)
notInExpr.setParseAction(NotInExpr)

inExpr = var + (pp.Suppress(lit("in")) + lbra + stringList + rbra)
inExpr.setParseAction(InExpr)

matchExpr = eqExpr | regexExpr | inExpr | notInExpr

booleanOp = lit("and") | lit("or")
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
        query: Filter expression (e.g., '@state = "DONE" and model = "bm25"')
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
