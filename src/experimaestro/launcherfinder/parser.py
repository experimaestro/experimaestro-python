# --- Parse specifications from strings

from functools import reduce
from arpeggio import (
    ZeroOrMore,
    OneOrMore,
    OrderedChoice,
    RegExMatch,
    ParserPython,
    StrMatch,
    Optional,
    PTNodeVisitor,
    visit_parse_tree,
    EndOfFile,
)
from . import specs

# --- Grammar


class SuppressStrMatch(StrMatch):
    suppress = True


def mem_spec():
    return "mem", "=", RegExMatch(r"\d+(GiB|MiB|G|M)?")


def cores_spec():
    return "cores", "=", RegExMatch(r"\d+")


def multiplier():
    return "*", RegExMatch(r"\d+")


def cuda_specs():
    return ZeroOrMore(OrderedChoice([mem_spec]), sep=",")


def cuda():
    return "cuda", "(", cuda_specs, ")", Optional(multiplier)


def cpu_specs():
    return ZeroOrMore(OrderedChoice([mem_spec, cores_spec]), sep=",")


def cpu():
    return "cpu", "(", cpu_specs, ")"


def duration():
    return (
        "duration",
        "=",
        RegExMatch(r"\d+"),
        RegExMatch(r"h(ours?)?|d(ays?)?|m(ins?)?"),
    )


def one_spec():
    return OneOrMore(OrderedChoice([duration, cuda, cpu]), sep="&")


def grammar():
    return OneOrMore(one_spec, sep="|"), EndOfFile()


# ---- Visitor


class Visitor(PTNodeVisitor):
    def visit_grammar(self, node, children):
        return specs.RequirementUnion(*[child for child in children])

    def visit_one_spec(self, node, children):
        return reduce(lambda x, el: x & el, children)

    def visit_duration(self, node, children):
        return specs.duration(" ".join(children))

    def visit_cuda(self, node, children):
        if len(children) > 1:
            return specs.cuda_gpu(**children[0]) * int(children[1])
        return specs.cuda_gpu(**children[0])

    def visit_cpu(self, node, children):
        return specs.cpu(**children[0])

    def visit_specs(self, node, children):
        a = children[0]
        for c in children[1:]:
            a.update(c)
        return a

    visit_cuda_specs = visit_specs
    visit_cpu_specs = visit_specs

    def visit_mem_spec(self, node, children):
        return {"mem": node.value}

    def visit_cores_spec(self, node, children):
        return {"cores": int(node.value)}


def parse(expr: str):
    """Parse a requirement specification string into a HostRequirement object.

    The specification string describes hardware requirements for running a task.
    Multiple alternatives can be specified using ``|`` (OR), and requirements
    within an alternative are combined using ``&`` (AND).

    **Syntax elements:**

    - ``duration=<N><unit>``: Job duration (units: h/hours, d/days, m/mins)
    - ``cpu(mem=<size>, cores=<N>)``: CPU requirements
    - ``cuda(mem=<size>) * <N>``: GPU requirements (memory and count)
    - Memory sizes: ``<N>G``, ``<N>GiB``, ``<N>M``, ``<N>MiB``

    :param expr: The requirement specification string
    :return: A :class:`~experimaestro.launcherfinder.specs.HostRequirement` object

    **Example:**

    .. code-block:: python

        from experimaestro.launcherfinder.parser import parse

        # Request 2 GPUs with 32GB each, 700GB RAM, for 40 hours
        # OR 4 GPUs with 32GB each for 50 hours
        req = parse(
            "duration=40h & cpu(mem=700GiB) & cuda(mem=32GiB) * 2"
            " | duration=50h & cpu(mem=700GiB) & cuda(mem=32GiB) * 4"
        )
    """
    parser = ParserPython(grammar, syntax_classes={"StrMatch": SuppressStrMatch})
    parse_tree = parser.parse(expr)
    return visit_parse_tree(parse_tree, Visitor(debug=False))
