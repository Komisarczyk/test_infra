#!/bin/sh
#needs to be in a directory containing polybench-c-4.2.1-beta
FILES=$(ls -d */)
for FILE in $FILES 
    do 
        FILE=${FILE//\/}
        echo "$FILE"
        cd $FILE
        if [ "$FILE" == "stencils" ]; then
            KERNELS=$(ls -d */)
                for KERNEL in $KERNELS 
                    do 
                        KERNEL=${KERNEL//\/}
                        echo "$KERNEL"
                        cd $KERNEL
                        {
                            rm *.mlir
                            mlir-pet -I ../../../polybench-c-4.2.1-beta/utilities -I ../../../polybench-c-4.2.1-beta/linear-algebra/$FILE/gemver -I ../../../polybench-c-4.2.1-beta/utilities/polybench.c ../../../polybench-c-4.2.1-beta/$FILE/$KERNEL/$KERNEL.c > $KERNEL.mlir
                            make clean                         
                            make all
                        } &> /dev/null
                        #clang-format-3.9 -i main.c
                        SCRIPT=$(ls *.out)
                        {
                            ./$SCRIPT
                            echo $?
                        }
                        cd ..
                done
        else
            KERNELS=$(ls -d */)
                for KERNEL in $KERNELS 
                    do 
                        KERNEL=${KERNEL//\/}
                        echo "$KERNEL"
                        cd $KERNEL
                        {
                            rm *.mlir
                            mlir-pet -I ../../../polybench-c-4.2.1-beta/utilities -I ../../../polybench-c-4.2.1-beta/linear-algebra/$FILE/gemver -I ../../../polybench-c-4.2.1-beta/utilities/polybench.c ../../../polybench-c-4.2.1-beta/linear-algebra/$FILE/$KERNEL/$KERNEL.c > $KERNEL.mlir
                            make clean  
                            make all
                        } &> /dev/null
                        #clang-format-3.9 -i main.c
                        SCRIPT=$(ls *.out)
                        {
                            ./$SCRIPT
                            echo $?
                        }
                        cd ..
                done
        fi
        cd ..
done