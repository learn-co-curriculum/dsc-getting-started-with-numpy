
# Introducing NumPy

NumPy is one of the main libraries for performing scientific computing in Python. Using NumPy, you can create high-performance multi-dimensional arrays, and several tools to work with these arrays. 

A numpy array can store a grid of values. All the values must be of the same type. numpy arrays are n-dimensional, and the number of dimensions is denoted the *rank* of the numpy array. The shape of an array is a tuple of integers which hold the size of the array along each of the dimensions.

For more information on numpy, we refer to http://www.numpy.org/.


## Objectives
* Know that numpy is imported as np: import numpy as np
* Understand how to initialize numpy arrays from nested Python lists, and access elements using square brackets
* Understand the shape attribute on numpy arrays
* Understand how to create arrays from scratch including Np.zeros, np.ones, np.full
* Learn how to do indexing in arrays
    * Slicing
    * Integer indexing
    * Boolean indexing
* Learn to perform scalar and vector math  

To work with NumPy arrays, you should import `numpy` first.  
**The naming convention for `numpy` is to import it as `np`.**


```python
import numpy as np
```

## Numpy array creation and basic operations
One easy way to create a numpy array is from a python list. The two are similar in a number of manners but NumPy is optimized in a number of ways for performing mathematical operations, including having a number of built in methods that will be extraordinarily useful.


```python
x = np.array([1,2,3])
print(type(x))
```

    <class 'numpy.ndarray'>


# Broadcasting Mathematical Operations

Notice right off the bat how basic mathematical operations will be applied element wise in a NumPy array versus a literal interpretation with a python list:


```python
x * 3 #multiplies each element by 3
```




    array([3, 6, 9])




```python
[1,2,3] * 3 #returns the list 3 times
```




    [1, 2, 3, 1, 2, 3, 1, 2, 3]




```python
x + 2 #Adds two to each element
```




    array([3, 4, 5])




```python
[1,2,3] + 2 # Returns an error; different data types
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-7-e2d4447f4589> in <module>()
    ----> 1 [1,2,3] + 2
    

    TypeError: can only concatenate list (not "int") to list


# Multidimensional Arrays
NumPy arrays are also very useful for storing multidimensional data such as matrices. Notice how NumPy tries to nicely allign the elements.


```python
#An ordinary nested list
y = [[1,2], [3,4]]
print(type(y))
y
```

    <class 'list'>





    [[1, 2], [3, 4]]




```python
#Reformatted as a NumPy array
y = np.array([[1,2], [3,4]])
print(type(y))
y
```

    <class 'numpy.ndarray'>





    array([[1, 2],
           [3, 4]])



# The Shape Attribute
One of the most important attributes to understand with this is the shape of a NumPy array.


```python
y.shape
```




    (2, 2)




```python
y = np.array([[1,2,3],[4,5,6]])
print(y.shape)
y
```

    (2, 3)





    array([[1, 2, 3],
           [4, 5, 6]])




```python
y = np.array([[1,2],[3,4],[5,6]])
print(y.shape)
y
```

    (3, 2)





    array([[1, 2],
           [3, 4],
           [5, 6]])



### We can also have higher dimensional data such as working with 3 dimensional data
<img src="3d_array2.png" width=500>


```python
y = np.array([[[1,2],[3,4],[5,6]],
             [[1,2],[3,4],[5,6]]
             ])
print(y.shape)
y
```

    (2, 3, 2)





    array([[[1, 2],
            [3, 4],
            [5, 6]],
    
           [[1, 2],
            [3, 4],
            [5, 6]]])



# Built in Methods for Creating Arrays
NumPy also has several built in methods for creating arrays that are useful in practice. In particular these methods are particularly useful:
* np.zeros(shape) 
* np.ones(shape)
* np.full(shape, fill)


```python
np.zeros(5) #one dimensional; 5 elements
```




    array([ 0.,  0.,  0.,  0.,  0.])




```python
np.zeros([2,2]) #two dimensional; 2x2 matrix
```




    array([[ 0.,  0.],
           [ 0.,  0.]])




```python
np.zeros([3,5]) #2 dimensional;  3x5 matrix
```




    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])




```python
np.zeros([3,4,5]) #3 dimensional; 3 4x5 matrices
```




    array([[[ 0.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0.]],
    
           [[ 0.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0.]],
    
           [[ 0.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0.]]])



##### Similarly the np.ones() method returns an array of ones


```python
np.ones(5)
```




    array([ 1.,  1.,  1.,  1.,  1.])




```python
np.ones([3,4])
```




    array([[ 1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.]])



##### The generalization of this is the np.full() method which allows you to create an array of arbitrary values.


```python
np.full(5, 3) #Create a 1d array with 5 elements, all of which are 3
```




    array([3, 3, 3, 3, 3])




```python
np.full(5, range(5)) #Create a 1d array with 5 elements, filling them with the values 0 to 4
```




    array([0, 1, 2, 3, 4])




```python
#Sadly this trick won't work for multidimensional arrays
np.full([2,5], range(10))
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-44-acc44367711c> in <module>()
          1 #Sadly this trick won't work for multidimensional arrays
    ----> 2 np.full([2,5], range(10))
    

    ~\Anaconda3wenv\lib\site-packages\numpy\core\numeric.py in full(shape, fill_value, dtype, order)
        301         dtype = array(fill_value).dtype
        302     a = empty(shape, dtype, order)
    --> 303     multiarray.copyto(a, fill_value, casting='unsafe')
        304     return a
        305 


    ValueError: could not broadcast input array from shape (10) into shape (2,5)



```python
np.full([2,5], np.pi) #NumPy also has useful built in mathematical numbers
```




    array([[ 3.14159265,  3.14159265,  3.14159265,  3.14159265,  3.14159265],
           [ 3.14159265,  3.14159265,  3.14159265,  3.14159265,  3.14159265]])



## Numpy array subsetting

You can subset NumPy arrays very similar to list slicing in python.


```python
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
print(x.shape)
x
```

    (4, 3)





    array([[ 1,  2,  3],
           [ 4,  5,  6],
           [ 7,  8,  9],
           [10, 11, 12]])




```python
x[0] #Retrieving the first row
```




    array([1, 2, 3])




```python
x[1:] #Retrieving all rows after the first row
```




    array([[ 4,  5,  6],
           [ 7,  8,  9],
           [10, 11, 12]])



### This becomes particularly useful in multidimensional arrays when we can slice on multiple dimensions


```python
#x[slice_dim1, slice_dim2]
x[:,0] #All rows, column 0
```




    array([ 1,  4,  7, 10])




```python
x[2:4,1:3] #Rows 2 through 4, columns 1 through 3
```




    array([[ 8,  9],
           [11, 12]])



### Notice that you can't slice in multiple dimensions naturally with built in lists


```python
x = [[1,2,3], [4,5,6], [7,8,9], [10,11,12]]
x

```




    [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]




```python
x[0]
```




    [1, 2, 3]




```python
x[:,0]
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-67-01770d3fab0a> in <module>()
    ----> 1 x[:,0]
    

    TypeError: list indices must be integers or slices, not tuple



```python
#To slice along a second dimension with lists we must verbosely use a list comprehension
[i[0] for i in x]
```




    [1, 4, 7, 10]




```python
#Doing this in multiple dimensions with lists
[i[1:3] for i in x[2:4]]
```




    [[8, 9], [11, 12]]



### 3D Slicing


```python
#With an array
x = np.array([
              [[1,2,3], [4,5,6]],
              [[7,8,9], [10,11,12]]
             ])
x
```




    array([[[ 1,  2,  3],
            [ 4,  5,  6]],
    
           [[ 7,  8,  9],
            [10, 11, 12]]])




```python
x.shape
```




    (2, 2, 3)




```python
x[:,:,-1]
```




    array([[ 3,  6],
           [ 9, 12]])



# Sources

http://cs231n.github.io/python-numpy-tutorial/#numpy

http://www.numpy.org/

https://campus.datacamp.com/courses/intro-to-python-for-data-science/chapter-1-python-basics?ex=1
