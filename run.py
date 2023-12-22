

import streamlit.web.cli as stcli
import os, sys
from Do_film import done1
from stream import done




if __name__ == "__main__":
    sys.argv = [
        "streamlit",
        "run",
        "stream.py",
        "--global.developmentMode=false",
    ]
    sys.exit(stcli.main())

done()
done1()



'''from pathlib import Path
from streamlit.web import bootstrap
import os

bootstrap.run(os.path.realpath('Do_film/stream.py'), '', [], [])'''