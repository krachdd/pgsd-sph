# copy Python modules to the build directory to make it a working Python package
MACRO(copy_file file)
    add_custom_command (
        OUTPUT ${file}
        DEPENDS ${file}
        POST_BUILD
        COMMAND    ${CMAKE_COMMAND}
        ARGS       -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file} ${CMAKE_CURRENT_BINARY_DIR}/${file}
        COMMENT    "Copy pgsd/test/${file}"
    )
ENDMACRO(copy_file)

# test_fl.py, test_hoomd.py and test_largefile.py are unmodified copies of the GSD 3.2
# tests (they import gsd, not pgsd) and are not run. The PGSD tests are in test_pgsd.py.
set(files conftest.py pytest.ini pgsd_testdata.py pgsd_mpi_writer.py test_pgsd.py)

foreach(file ${files})
    copy_file(${file})
endforeach()

add_custom_target(copy_pgsd_test ALL DEPENDS ${files})
