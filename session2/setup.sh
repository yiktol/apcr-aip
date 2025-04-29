#!/usr/bin/bash

directory=~/environment/apcr-aip/session2

cd $directory

if [ -d $directory/.env ]; 
    then
        echo "Directory exists."
        source $directory/.env/bin/activate
    else
        echo "Directory does not exists."

        echo "Creating Virtual Environment"
        python -m venv .env
        source $directory/.env/bin/activate

        echo "Installing dependencies"
        pip install -U pip
        pip install -r requirements.txt
        
fi

echo "Starting Application"
$directory/.env/bin/streamlit run $directory/Home.py --server.port 8082 &

deactivate