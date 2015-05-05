if [ -f build ];
then
    rm -r build
fi;

mkdir build
javac -classpath weka.jar -d build diffprivacy/*.java
java  -classpath build:weka.jar diffprivacy.TestExpForest
