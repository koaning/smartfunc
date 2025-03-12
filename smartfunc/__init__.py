class backend:
    def __init__(self, name, system=None, cache=None, **kwargs):
        self.model = llm.get_model(name)
        self.system = system
        self.kwargs = kwargs
        self.cache = Cache(cache) if isinstance(cache, str) else cache

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            signature = inspect.signature(func)
            docstring = inspect.getdoc(func) or ""
            type_hints = get_type_hints(func)

            # We only support Pydantic now
            if type_hints.get('return', None):
                assert issubclass(type_hints.get('return', None), BaseModel), "Output type must be Pydantic class"
        
            # Create a dictionary of parameter types
            param_types = {name: param.default for name, param in signature.parameters.items()}
            bound_args = signature.bind(*args, **kwargs)
            bound_args.apply_defaults()  # Apply default values for missing parameters
            all_kwargs = bound_args.arguments
        
            template = Template(docstring)
            formatted_docstring = template.render(**all_kwargs)
            cache_key = docstring + json.dumps(all_kwargs) + str(type_hints.get('return', None))
        
            if self.cache:
                if cache_key in self.cache:
                    return self.cache[cache_key]
        
            # Call the prompt, with schema if given
            resp = self.model.prompt(
                formatted_docstring, 
                system=self.system,
                schema=type_hints.get('return', None),
                **kwargs
            )
            if type_hints.get('return', None):
                out = json.loads(resp.text())
            out = resp.text()

            if self.cache:
                self.cache[cache_key] = out
            return out

        return wrapper

    def run(self, func, *args, **kwargs):
        new_func = self(func)
        return new_func(*args, **kwargs)

class async_backend:
    def __init__(self, name, system=None, **kwargs):
        self.model = llm.get_async_model(name)
        self.system = system
        self.kwargs = kwargs

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            signature = inspect.signature(func)
            docstring = inspect.getdoc(func) or ""
            type_hints = get_type_hints(func)

            # We only support Pydantic now
            if type_hints.get('return', None):
                assert issubclass(type_hints.get('return', None), BaseModel), "Output type must be Pydantic class"
        
            # Create a dictionary of parameter types
            param_types = {name: param.default for name, param in signature.parameters.items()}
            bound_args = signature.bind(*args, **kwargs)
            bound_args.apply_defaults()  # Apply default values for missing parameters
            all_kwargs = bound_args.arguments
        
            template = Template(docstring)
            formatted_docstring = template.render(**all_kwargs)
        
            # Call the prompt, with schema if given
            resp = await self.model.prompt(
                formatted_docstring, 
                system=self.system,
                schema=type_hints.get('return', None),
                **kwargs
            )
            text = await resp.text()
            if type_hints.get('return', None):
                return json.loads(text)
            return text

        return wrapper

    async def run(self, func, *args, **kwargs):
        new_func = self(func)
        return new_func(*args, **kwargs)