import globals from "globals";

export default [
    {
        files: ["widget/**/*.js"],
        languageOptions: {
            ecmaVersion: 2020,
            sourceType: "script",
            globals: {
                ...globals.browser,
                gtag: "readonly",
            },
        },
        rules: {
            "no-unused-vars": ["warn", { argsIgnorePattern: "^_", caughtErrorsIgnorePattern: "^_" }],
            "no-undef": "error",
            eqeqeq: "error",
            "no-implicit-globals": "error",
            indent: ["error", 4],
        },
    },
];
