clear

mkdir ./output
clang++ ./src/*.c++ -fsanitize=address -Wall -Werror -std=c++23 -fsanitize=undefined -O3 -o ./output/out

if [[ $? -ne 0 ]]; then

    printf "\n\n[SHELL] Compilation failed\n\n"
    exit -1 

fi

printf "\n\n[SHELL] Compilation success\n\n"
./output/out


printf "\n\n[SHELL] Program exit code $?\n\n"



