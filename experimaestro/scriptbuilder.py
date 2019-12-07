from pathlib import Path
from .scheduler import Workspace
from .connectors import Connector
from .commandline import CommandLineJob

class ShScriptBuilder:
  def __init__(self, shpath: Path):
    self.shpath = shpath


  def write(ws: Workspace, connector: Connector, path: Path, job: CommandLineJob):
    """Write the script file
    
    Arguments:
        ws {Workspace} -- The workspace
        connector {Connector} -- [description]
        path {Path} -- [description]
        job {CommandLineJob} -- [description]
    
    Returns:
        [type] -- [description]
    """
    assert isinstance(job, CommandLineJob)

    directory = path.parent
    scriptpath = directory / path.name.with_suffix(".sh")
    donepath = job.donepath
    startlockPath = job.pathTo(LOCK_START_PATH)
    exitcodepath = job.pathTo(EXIT_CODE_PATH)
  
    pidFile = job.pathTo(PID_PATH)

    LOGGER.info("Writing script %s", scriptpath)
    with scriptpath.open("wt") as out:
      out.write("#!{}\n", shpath)
      out.write("# Experimaestro generated task\n")

      # Output tags
      out.write("# __tags__ = ")
      YAML.Emitter yout(out)
      yout.SetMapFormat(YAML.Flow)
      yout 
      out.write(YAML.BeginMap)
      for item in job.parameters().tags(): 
        out.write(YAML.Key )
        out.write(item.first )
        out.write(YAML.Value )
        out.write(item.second.toYAML())

      yout 
      out.write(YAML.EndMap)
      out.write(std.endl )
      out.write("\n")

      context = CommandContext(ws, connector, path.parent, path.name)
      context.parameters = job.parameters()

      # --- Checks locks right away

      # Checks the main lock - if not there, are not protected
      if lockFiles:
        out.write("# Checks that the locks are set")
        out.write("\n")
        for lockFile in lockFiles:
          out.write("if not  test -f ")
          out.write(connector.resolve(lockFile) )         
          out.write("; then echo Locks not set; exit 017; fi")
          out.write("\n")

      # Checks the start lock to avoid two experimaestro launched processes to
      # start
      out.write("# Checks that the start lock is set, removes it")
      out.write("\n")
      out.write("if not  test -f ")
      out.write(connector.resolve(startlockPath))
      out.write("; then echo start lock not set; exit 017; fi")
      out.write("\n")
      out.write("rm -f ")
      out.write(connector.resolve(startlockPath) )
      out.write("\n")
      out.write("\n")

      # Use pipefail for fine grained analysis of errors in commands
      out.write("set -o pipefail")
      out.write(std.endl )
      out.write("\n")

      out.write("echo $$ > \"")
      out.write(protect_quoted(connector.resolve(pidFile)) )
      out.write("\"\n\n")

      for pair in environment.items():          
        out.write("export ")
        out.write(pair.first)
        out.write("=\"")
        out.write(protect_quoted(pair.second))           
        out.write("\"")
        out.write("\n")


      # Adds notification URL to script
      if not notificationURL.empty(): 
        out.write("export XPM_NOTIFICATION_URL=\"")        
        out.write(protect_quoted(notificationURL) )
        out.write("/")
        out.write(job.getJobId() )
        out.write("\"")
        
        out.write("\n")


      out.write("cd \"")
      out.write(protect_quoted(connector.resolve(directory)) )
      out.write("\"")
      out.write("\n")

      # Write some command
      if preprocessCommands:    
        preprocessCommands.output(context, out)

      # --- CLEANUP

      out.write("cleanup() {")
      out.write("\n")
      # Write something
      out.write(" echo Cleaning up 1>&2")
      out.write("\n")
      # Remove traps
      out.write(" trap - 0")
      out.write("\n")

      out.write(" rm -f ")
      out.write(connector.resolve(pidFile, directory) )
      out.write(";")
      out.write("\n")

      # Remove locks
      for file in lockFiles:
        out.write(" rm -f ")
        out.write(connector.resolve(file, directory) )
        out.write("\n")


      # Remove temporary files
      def cleanup(c):
        namedRedirections = context.getNamedRedirections(c, False)
        for file in namedRedirections.outputRedirections:
          out.write(" rm -f ")
          out.write(connector.resolve(file, directory) )
          out.write(";")                
          out.write("\n")

        for file in namedRedirections.errorRedirections:
          out.write(" rm -f ")
          out.write(connector.resolve(file, directory) )
          out.write(";")
          out.write("\n")

      command.forEach(cleanup)

      # Notify if possible
      if not notificationURL.empty():
        out.write(" wget --tries=1 --connect-timeout=1 --read-timeout=1 --quiet -O ")
        out.write("/dev/null \"$XPM_NOTIFICATION_URL?status=eoj\"")
        out.write("\n")


      # Kills remaining processes
      out.write(" test not  -z \"$PID\" and pkill -KILL -P $PID")
      out.write("\n")

      out.write("}")
      out.write(std.endl )
      out.write("\n")

      # --- END CLEANUP

      out.write("# Set trap to cleanup when exiting")
      out.write("\n")
      out.write("trap cleanup 0")
      out.write("\n")


      out.write("\n")
        
      out.write("""checkerror()  { local e; for e in \"$@\"; do [[ \"$e\" != 0 ]] and
          [[ 
          "$e" != 141 ]] and exit $e; done; return 0; }""")
        
      out.write(std.endl )
      out.write("\n")

      # The prepare all the command
      out.write("(")
      out.write("\n")
      command.output(context, out)

      out.write(") ")

      # Retrieve PID
      out.write(" & ")
      out.write("\n")
      out.write("PID=$not ")
      out.write("\n")
      out.write("wait $PID")
      out.write("\n")
      out.write("code=$?")
      out.write("\n")
      out.write("if test $code -ne 0; then")
      out.write("\n")
      out.write(" echo $code > \"")
          
      out.write(protect_quoted(connector.resolve(exitcodepath, directory)) )
      out.write("\"")
      
      out.write("\n")

      out.write(" exit $code")
      out.write("\n")
      out.write("fi")
      out.write("\n")
      
      out.write("echo 0 > \"")
      out.write(protect_quoted(connector.resolve(exitcodepath, directory)))      
      out.write("\"")
      out.write("\n")

      out.write("touch \"")
      out.write(protect_quoted(connector.resolve(donepath, directory)) )
      out.write("\"")
      out.write("\n")

    # Set the file as executable
    connector.setExecutable(scriptpath, True)
    return scriptpath
