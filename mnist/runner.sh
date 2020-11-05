for file in $(ls | grep -F .py)
do
    if [ $file != 'mnistloader.py' ] && [ $file != 'utils.py' ]
    then
        echo "$file started."
        time python3 $file
        echo "$file done."
    fi
done