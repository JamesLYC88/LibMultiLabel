Getting Started
===============

The library API is used in your own scripts if you want
more fine grained control over the training/prediction process.
For an application usable out-of-the-box, see the
`command line interface <../cli/linear.rst>`_ instead.

The library API is composed of a `linear classifier module <linear.html>`_ and a `neural network classifier module <nn.html>`_::

    import libmultilabel.nn
    import libmultilabel.linear

Both of which can be used independently.

Installation
^^^^^^^^^^^^

We provide two installation types to install LibMultiLabel:

* Install both neural network module and linear classifier module. ::

    pip3 install libmultilabel

* Install only linear classifier module without any torch-related requirements. ::

    pip3 install libmultilabel[linear]

Quickstart Tutorials
^^^^^^^^^^^^^^^^^^^^

These tutorials shows how the library API may be used in
end-to-end examples. After these tutorials, more in-depth
explanations can be found in the `user guides <placeholder>`_.

* `Linear Classification <linear_tutorial.html>`_
* `Neural Network <nn_tutorial.html>`_


