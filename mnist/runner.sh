for file in $(ls | grep -F .py)
do
    if [ $file != 'mnistloader.py' ] && [ $file != 'utils.py' ]
    then
        python3 $file
    fi
done