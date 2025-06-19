clear
clear
clear
clear
clear

mkdir ./output
#clang++ ./src/*.c++ -fsanitize=address -Wall -std=c++17 -fsanitize=undefined -O3 -o ./output/out
clang++ ./src/*.c++ -O3 -std=c++17 -ffast-math -march=native -flto=thin -pthread -DNDEBUG -o ./output/out_fast


if [[ $? -ne 0 ]]; then

    printf "\n\n[SHELL] Compilation failed\n\n"
    exit -1 

fi

printf "\n\n[SHELL] Compilation success\n\n"

#./output/out
./output/out_fast



python3 ./src/plot2D.py

printf "\n\n[SHELL] Program exit code $?\n\n"



