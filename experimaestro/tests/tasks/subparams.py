from experimaestro import task, param, subparam
import time


@param("x", type=int)
@subparam("epoch", type=int)
@task()
class Task:
    def execute(self):
        (self.__maintaskdir__ / f"{self.epoch}").write_text(str(self.x))
        print(time.time())
        time.sleep(0.1)
        print(time.time())
