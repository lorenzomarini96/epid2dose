Quick Start
===========

Example dataset
---------------

The repository contains a small example dataset under

.. code-block:: text

   examples/

       input/
       reference/
       output/

Running inference
-----------------

Predict Portal Dose images

.. code-block:: bash

   python pd_prediction.py \
       -i examples/input \
       -o examples/output

Generate a validation report

.. code-block:: bash

   python pd_prediction.py \
       -i examples/input \
       -o examples/output \
       --reference_dir examples/reference \
       --report

Output
------

The software automatically creates

.. code-block:: text

   output/

       predictions/
           PD_*.npy

       reports/
           prediction_overview.pdf
           validation_report.pdf