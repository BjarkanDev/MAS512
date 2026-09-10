import tensorflow as tf

def question1():
    
    a=tf.constant([
        [4.0, 5.0],
        [1.0, 2.0]
    ])

    b=tf.constant([
        [2.0, 3.0],
        [7.0, 8.0]
    ])

    ab_add=tf.add(a,b)
    ab_subtract=tf.subtract(a,b)
    ab_matmul=tf.matmul(a,b)
    ab_concat=tf.concat([a,b], axis=0)
    print(f'additon of matrix a = \n{a} and  \n b= \n{b} is \n ab_concat=\n{ab_concat}')


