echo
echo "this script can run the program through gitbash for windows, or unix terminal for OSX/Linux. It attempts to run through windows first"

echo "checking for pandas,numpy, scipy, and scikit"
pip3 install pandas
pip3 install numpy
pip3 install scipy
pip3 install scikit-learn
pip3 install nltk

echo
echo "trying to run python program, gitbash for windows style"
echo
if py code/main.py 
    then #windows running through git bash can execute on 3.7 as such
    echo
    echo "Program complete"
    exit
else echo "must be a unix system"
fi
echo

echo "trying to run python program unix style"
echo
if python3 code/main.py
    then #windows running through git bash can execute on 3.7 as such
    echo
    echo "Program complete"
    exit
else 
    echo "excuse me?"
fi