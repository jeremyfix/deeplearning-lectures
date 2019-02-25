---
title:  'HowTo pandoc markdown'
author:
- Jeremy Fix
keywords: [Pandoc markdown, howto]
...


Some options and extensions are provided by the makefile; Some metadata are inserted within the compilation line.

# Links

## Href

Inline link : [here is a doc of the pandoc MD](https://pandoc.org/MANUAL.html#pandocs-markdown)

## Biblio

You can cite a bibentry [@Lin2014]. These are automatically inserted at the end of the document. These entries are extracted and formated from the [biblio.bib](biblio.bib) file. It makes use of the pandoc citeproc extension.


# Codes 

## Python

~~~ {.sourceCode .python .numberLines}
import keras

def my_function():
   "just a test"
   print 8/2x
while(True):
   x = x + 1
print(x)
~~~

Ok

``` {.sourceCode .python}
def my_function2():
   "just a test dlkdjqlkdsj"
   print 8/2x
```

## Bash 

``` console
mymachine:~:mylogin$ echo "Das ist rishtig wunderbach"
Das ist rishtig wunderbach
```

## Math

Below is an example of a math block

$$\xi = 2$$

$$\hat{\xi} = \frac{1}{p}\sum_{i=1}^{n} \sum_{x_j \in C_i} \lVert
 x_j-\mathbf{w}_i \rVert^2.$$

But you could also make use of inline math as in $e^{i\pi} + 1 = 0$.

## References
