from experimaestro import *

hw = Typename("helloworld")

@Argument("word", type=str, required=True, help="Word to generate")
@Task(hw.say)
class Say:
    def execute(self):
        print(self.word.upper(),)

@Argument("strings", type=Array(Say), help="Strings to concat")
@Task(hw.concat)
class Concat:
    def execute(self):
        # We access the file where standard output was stored
        says = []
        for string in self.strings:
            with open(string._stdout()) as fp:
                says.append(fp.read().strip())
        print(" ".join(says))

if __name__ == "__main__":
    register.parse()
