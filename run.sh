python setup.py install

sh train.sh  > train.log  2>&1 &
tail -f train.log

shutdown