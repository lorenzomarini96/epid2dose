Installation
============

Clone the repository

.. code-block:: bash

   git clone https://github.com/lorenzomarini96/epid2dose.git

   cd epid2dose

Create a virtual environment (recommended)

.. code-block:: bash

   python -m venv .venv

Activate it

Linux / macOS

.. code-block:: bash

   source .venv/bin/activate

Windows

.. code-block:: bat

   .venv\Scripts\activate

Install the required packages

.. code-block:: bash

   pip install -r requirements.txt

Download trained weights
------------------------

The trained network weights are distributed separately because they exceed
GitHub file size limits.

Download the weights from

https://drive.google.com/drive/folders/1kV3Vw6ueXHsnLeENCj65CruUSFxBfPMC

and extract them inside

.. code-block:: text

   models/
   └── weights/
       └── cv_1/
           ├── model_1/
           ├── model_2/
           ├── model_3/
           ├── model_4/
           └── model_5/