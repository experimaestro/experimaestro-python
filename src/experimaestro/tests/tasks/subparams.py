from experimaestro import task, Param, SubParam
import time


@task()
class Task:
    epoch: SubParam[int]
    x: Param[int]

    def execute(self):
        (self.__maintaskdir__ / f"{self.epoch}").write_text(str(self.x))
        print(time.time())
        time.sleep(0.1)
        print(time.time())
