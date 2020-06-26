// https://docs.python.org/3/extending/extending.html

#include <Python.h>

static PyObject *method_echo(PyObject *self, PyObject *args)
{
  char *msg;

  // Parse arguments
  if (!PyArg_ParseTuple(args, "ss", &msg))
  {
    return NULL;
  }

  printf("You message was: %s", msg);
  Py_RETURN_NONE;
}

static PyMethodDef ExampleMethods[] = {
    {"echo", method_echo, METH_VARARGS, "Python interface for printf C library function"},
    {NULL, NULL, 0, NULL} /* Sentinel value */
};

static struct PyModuleDef segmentation_module = {
  PyModuleDef_HEAD_INIT,
  "echo",
  "Python interface for printf C library function",
  -1,
  ExampleMethods
};
