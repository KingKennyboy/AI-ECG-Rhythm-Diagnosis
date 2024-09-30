#!/bin/bash

export PATH="/var/lib/jenkins/.local/bin:$PATH"
export NVM_DIR="/var/lib/jenkins/.nvm"


echo "Starting build process..."
. $NVM_DIR/nvm.sh
nvm install 21.0.0

if ! command -v npm &> /dev/null; then
    echo "npm is not installed. Installing Node.js and npm..."
    nvm install node

    nvm use node
else
    echo "npm is already installed."
fi

if ! command -v pytest &> /dev/null; then
    echo "pytest not found, installing..."
    pip install --user pytest
fi

pip install --user pytest coverage pytest-cov

echo "Checking Dependencies..."
if ! pip freeze | grep -q -f ../requirements.txt; then
    echo "Installing Dependencies..."
    pip install --user -r ../requirements.txt
else
    echo "Dependencies are already installed."
fi

echo "Running pytest for Python testing and code coverage..."

pytest -v --cov=../test ../test
if [ $? -ne 0 ]; then
 echo "Tests failed"
    exit 1
else
    echo "Tests passed successfully"
fi



#if ! command -v pylint &> /dev/null; then
    #echo "pylint not found, installing..."
   # pip install --user pylint
#fi

#echo "Running Pylint for Python linting..."

#FILES_TO_ANALYZE="../trainingModel"

#pylint $FILES_TO_ANALYZE
#if ! command -v npm &> /dev/null; then
   #echo "npm not found, installing..."
   #npm install
#fi

if ! command -v jest &> /dev/null; then
    echo "Jest is not installed, installing jest..."
    npm install -g jest

else
    echo "Jest is installed. "
fi

npm install @babel/preset-env --save-dev

echo "Running jest for JavaScript testing and code coverage..."

npm run test:coverage

if ! command -v eslint &> /dev/null; then
    echo "ESlint is not installed, installing jest..."
    npm install -g eslint

else
    echo "ESlint is installed. "
fi


#echo "Running ESLint for JavaScript linting..."

#npm install --save-dev eslint-config-prettier

#npm run lint

