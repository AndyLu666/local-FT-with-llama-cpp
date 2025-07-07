set(TARGET llama-logistic-regression)
add_executable(${TARGET} logistic-regression.cpp)
install(TARGETS ${TARGET} RUNTIME)
# 不依賴 llama 庫，僅需 pthreads
find_package(Threads REQUIRED)
target_link_libraries(${TARGET} PRIVATE Threads::Threads)
target_compile_features(${TARGET} PRIVATE cxx_std_17) 