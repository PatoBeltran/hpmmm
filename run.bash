#!/bin/bash
echo "" > mmm_output.log

I=0
echo Test Fastest >> mmm_output.log
./mmm 4096 >> mmm_output.log
while [ $I -lt 10 ]; do
  ./mmm $(((RANDOM % 5000)+1)) >> mmm_output.log
  let I=I+1
done
echo ----- >> mmm_output.log
I=0
echo Test Data Copying >> mmm_output.log
./mmmd 4096 >> mmm_output.log
while [ $I -lt 10 ]; do
  ./mmmd $(((RANDOM % 5000)+1)) >> mmm_output.log
  let I=I+1
done
echo ----- >> mmm_output.log
I=0
echo Test Cache Blocking >> mmm_output.log
./mmmc 4096 >> mmm_output.log
while [ $I -lt 10 ]; do
  ./mmmc $(((RANDOM % 5000)+1)) >> mmm_output.log
  let I=I+1
done
echo ----- >> mmm_output.log
I=0
echo Test Register Blocking >> mmm_output.log
./mmmr 4096 >> mmm_output.log
while [ $I -lt 10 ]; do
  ./mmmr $(((RANDOM % 5000)+1)) >> mmm_output.log
  let I=I+1
done
echo ----- >> mmm_output.log
