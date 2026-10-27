"""Execute project Python tools without inheriting shell credentials/configuration."""
import os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from service_runner import environment,ROOT
py=str(ROOT/'runtime/app-env/bin/python')
os.chdir(ROOT)
os.execve(py,[py,*sys.argv[1:]],environment())
