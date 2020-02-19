function metric_steepest_descent!()
    return nothing
end

function metric_projection(::SteepestDescent, W, D;
    ρ_init::Real      = 1.0,
    maxiters::Integer = 100,
    penalty::Function = __default_schedule,
    history::FuncLike = __default_logger) where FuncLike
    #
    return nothing
end
