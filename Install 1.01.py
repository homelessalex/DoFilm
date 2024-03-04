import os
import pathlib


os.system(  "cd " + os.path.dirname(os.path.abspath(__file__))+"\n"+
            "git clone https://github.com/homelessalex/DoFilm\n"+
            "cd dofilm\n"+
            "mkdir rendered imgs\n"+
            "python3.12 -m venv venv\n" +
            "source venv/bin/activate\n"+
            "pip3 install numpy colour-science Cython setuptools flet matplotlib numba scipy wheel rawpy py2app\n"+
            "cd " + os.path.abspath("DoFilm")+"\n"
            "python3 setup.py py2app\n")

os.system(  "mv " + os.path.abspath("DoFilm/dist/DoFilm.app") + " " + os.path.dirname(os.path.abspath(__file__))+"\n" +
            "cd " + os.path.dirname(os.path.abspath(__file__))+"\n"+
            "rm " + "-rf dist"  +"\n"+
            "rm " + "-rf " + "build" + "\n" +
            "rm " + "-rf " + "DoFilm" + "\n")
#os.replace(os.path.abspath("dist/DoFilm.app"), os.path.abspath("dist/DoFilm.app"))
#os.remove(os.path.abspath("dist"))
#os.remove(os.path.abspath("build"))
#"mv " + os.path.abspath("dist/DoFilm.app") + " " + os.path.dirname(os.path.abspath(__file__))+"\n" +
