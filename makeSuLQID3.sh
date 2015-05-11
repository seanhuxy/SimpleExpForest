#!/bin/bash
if [ -f build ];
then
    rm -r build
fi;

package=technion.cs
path=technion/cs

mkdir build
javac -classpath weka.jar -d build  ${path}/*.java \
                                    ${path}/PrivacyAgents/*.java \
                                    ${path}/Scorer/*.java \
                                    ${path}/test/*.java 

for depth in $(seq 4 3 10)
do
    for e in $(seq 1 3 10)
    do
        java  -classpath build:weka.jar ${package}.test.TestSuLQID3 $depth $e
    done
done


