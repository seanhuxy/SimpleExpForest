if [ -f build ];
then
    rm -r build
fi;

mkdir build
javac -classpath weka.jar -d build diffprivacy/*.java

for depth in $(seq 4 3 10)
do
    for e in $(seq 1 3 10)
    do
        java  -classpath build:weka.jar diffprivacy.TestExpForest $depth $e
        #java  -classpath build:weka.jar ${package}.test.TestSuLQID3 $depth $e
    done
done


