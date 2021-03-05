# Introduction

Experiment managers are conceptually linked to job scheduling software
such as cluster-based [OAR]() or [Slurm](). Those tools however do
not target experiment management, and are thus orthogonal to our
purpose. There are projects closer to our work, namely [Comet](),
[Sacred](), [FGLab](), [Sumatra]() that all track down
experimental parameters. Comet has a strong focus on collaboration and
note taking, but targets machine learning single shot experiments and is
not open source. Sumatra and FGLab are based on parameter files and are
less flexible. The closest to our work, Sacred, is an open-source
project that allows to have pre-processing steps (the ingredients), but
there is no way to build complex experimental plans as in Experimaestro.
More precisely, Sacred and all other experiment managers (as far as we
know) targets a single run of an experimental pipeline rather than
managing a set of related experimental tasks. They all consider that an
experimental plan is declarative – typically defined as a set of
parameter files, but this turns out to make things complicated when
building complex experimental plans.

Compared to those projects, Experimaestro has three distinctive
features. More precisely, it

1. defines types and tasks that can be
   composed within an experimental plan,

1. has a clear way to indicate
   which experimental parameters are monitored through the use of tags,

1. automatically organizes tasks outputs within the file system, removing
   the burden of choosing where to store a task result, and

1. most
   importantly, it defines experiments imperatively and not declaratively.

!!! note
Experimaestro and datamaestro are described in the following paper

    Benjamin Piwowarski. 2020.
    [Experimaestro and Datamaestro: Experiment and Dataset Managers (for IR).](https://doi.org/10.1145/3397271.3401410)
    *In Proceedings of the 43rd International ACM SIGIR*
