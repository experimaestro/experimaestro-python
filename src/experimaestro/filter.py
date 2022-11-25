from typing import Any, Callable, Dict, List, Optional
import pyparsing as pp
from pathlib import Path
import json
from experimaestro.compat import cached_property
import regex
from experimaestro.scheduler import JobState


class JobInformation:
    def __init__(self, path: Path, scriptname: str):
        self.path = path
        self.scriptname = scriptname

    @cached_property
    def params(self):
        return json.loads((self.path / "params.json").read_text())

    @cached_property
    def tags(self) -> List[str]:
        return self.params["tags"]

    @cached_property
    def state(self) -> Optional[JobState]:
        if (self.path / f"{self.scriptname}.pid").is_file():
            return JobState.RUNNING
        elif (self.path / f"{self.scriptname}.done").is_file():
            return JobState.DONE
        elif (self.path / f"{self.scriptname}.failed").is_file():
            return JobState.ERROR
        else:
            return None

    def getprocess(self):
        from experimaestro.connectors import Process
        from experimaestro.connectors.local import LocalConnector

        connector = LocalConnector.instance()
        pinfo = json.loads((self.path / f"{self.scriptname}.pid").read_text())
        return Process.fromDefinition(connector, pinfo)


# --- classes for processing


class VarExpr:
    def __init__(self, values):
        (self.varname,) = values

    def get(self, info: JobInformation):
        if self.varname == "@state":
            return info.state.name if info.state else None

        return info.tags.get(self.varname, None)

    def __repr__(self):
        return f"""VAR<{self.varname}>"""


class BaseInExpr:
    def __init__(self, values):
        self.var, *stringList = values
        self.values = set(stringList)


class InExpr(BaseInExpr):
    def filter(self, information: JobInformation):
        value = self.var.get(information)
        return value in self.values

    def __repr__(self):
        return f"""IN<{self.varname}|{",".join(self.values)}>"""


class NotInExpr(BaseInExpr):
    def filter(self, information: JobInformation):
        value = self.var.get(information)
        return value not in self.values

    def __repr__(self):
        return f"""NOT_IN<{self.varname}|{",".join(self.values)}>"""


class RegexExpr:
    def __init__(self, tokens):
        self.var, expr = tokens
        self.regex = regex.compile(expr)

    def __repr__(self):
        return f"""REGEX[{self.varname}, {self.value}]"""

    def matches(self, manager, publication):
        if self.varname == "tag":
            return self.value in publication.tags

        raise AssertionError()

    def filter(self, information: JobInformation):
        value = self.var.get(information)
        if not value:
            return False

        return self.regex.match(value)


class ConstantString:
    def __init__(self, tokens):
        (self.value,) = tokens

    def get(self, information: JobInformation):
        return self.value

    def __repr__(self):
        return f'"{self.value}"'


class EqExpr:
    def __init__(self, tokens):
        self.var1, self.var2 = tokens

    def __repr__(self):
        return f"""EQ[{self.var1}, {self.var2}]"""

    def filter(self, information: JobInformation):
        return self.var1.get(information) == self.var2.get(information)


class LogicExpr:
    """Logical expression"""

    def __init__(self, tokens):
        self.operator, self.y = tokens
        self.x = None

    def filter(self, information: JobInformation):
        if self.operator == "and":
            return self.y.filter(information) and self.x.filter(information)

        print(self.y.filter(information), "OR", self.x.filter(information))

        return self.y.filter(information) or self.x.filter(information)

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

l = pp.Literal

lpar, rpar, lbra, rbra, eq, comma, pipe, tilde = map(pp.Suppress, "()[]=,|~")
quotedString = pp.QuotedString('"', unquoteResults=True) | pp.QuotedString(
    "'", unquoteResults=True
)

var = l("@state") | pp.Word(pp.alphas)
var.setParseAction(VarExpr)

regexExpr = var + tilde + quotedString
regexExpr.setParseAction(RegexExpr)

varQuotedString = quotedString
varQuotedString.setParseAction(ConstantString)
eqExpr = var + eq + (var | quotedString)
eqExpr.setParseAction(EqExpr)

stringList = quotedString + pp.ZeroOrMore(comma + quotedString)

notInExpr = var + (pp.Suppress(l("not in")) + lbra + stringList + rbra)
notInExpr.setParseAction(NotInExpr)

inExpr = var + (pp.Suppress(l("in")) + lbra + stringList + rbra)
inExpr.setParseAction(InExpr)

matchExpr = eqExpr | regexExpr | inExpr | notInExpr

booleanOp = l("and") | l("or")
logicExpr = (
    matchExpr + pp.ZeroOrMore((booleanOp + matchExpr).setParseAction(LogicExpr))
).setParseAction(LogicExpr.summary)
parenExpr = logicExpr | (lpar + logicExpr + rpar)

filterExpr = (
    parenExpr + pp.ZeroOrMore((booleanOp + parenExpr).setParseAction(LogicExpr))
).setParseAction(LogicExpr.summary)
expr = (matchExpr + pp.Optional(pipe + filterExpr)).setParseAction(LogicExpr.generator)


def createFilter(query: str) -> Callable[[Dict[str, Any]], bool]:
    """Returns a filter object given a query"""
    (r,) = logicExpr.parseString(query, parseAll=True)
    return r.filter
