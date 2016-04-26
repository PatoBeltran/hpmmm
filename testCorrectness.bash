#!/bin/bash 
INITIAL_SIZE=1
FINAL_SIZE=400

echo Start test. Testing from $INITIAL_SIZE to $FINAL_SIZE
let COUNTER=INITIAL_SIZE
while [ $COUNTER -lt $FINAL_SIZE ]; do
  ./mmm $COUNTER -t

  if ! cmp mine.txt correct.txt >/dev/null 2>&1
  then
    echo Error at size $COUNTER
  fi

  let COUNTER=COUNTER+1 
done
echo End test.
