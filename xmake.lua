add_languages("c++17")

add_rules("mode.release")
local depends = {
    "eigen", "stb", "nlohmann_json"
}
add_requires(depends)


target("NGP-Simulator")
    add_packages(depends, {public = true})
    set_kind("static")

    add_includedirs({
        "Modules/Camera",
        "Modules/HashEncoding",
        "Modules/MLP",
        "Modules/SHEncoding",
        "Utils/",
        "Utils/Image"
        }, {public = true}
    )
    add_files({
        "Modules/Camera/*.cpp",
        "Modules/HashEncoding/*.cpp",
        "Modules/MLP/*.cpp",
        "Utils/Image/image.cpp",
        "Modules/SHEncoding/*.cpp"
    })
    add_files("NGP_Simulator.cpp")

target("NGP-Sim")
    set_kind("binary")
    add_files("main.cpp")

    add_deps("NGP-Simulator")

    set_targetdir(".")

target("QModule")
    add_packages(depends, {public = true})
    set_kind("static")

    add_includedirs({
        "QModules/Camera",
        "QModules/HashEncoding",
        "QModules/MLP",
        "QModules/SHEncoding",
        "Utils/",
        "Utils/Image"
        }, {public = true}
    )
    add_files({
        "QModules/Camera/*.cpp",
        "QModules/HashEncoding/*.cpp",
        "QModules/MLP/*.cpp",
        "Utils/Image/image.cpp",
        "QModules/SHEncoding/*.cpp"
    })

target("QTest")
    add_deps("QModule")
    set_kind("binary")
    add_files("QModuleTest.cpp")
    set_targetdir(".")