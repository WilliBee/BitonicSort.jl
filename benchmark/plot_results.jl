using CairoMakie
using CSV
using DataFrames
using Statistics

# Settings
const INPUT_CSV = joinpath("benchmark", "benchmark_results.csv")
const OUTPUT_DIR = joinpath("benchmark", "plots")

# Set theme
set_theme!(theme_latexfonts())

# Create output directory
mkpath(OUTPUT_DIR)

println("Loading results from $INPUT_CSV...")
df = CSV.read(INPUT_CSV, DataFrame)

# Filter out failed runs
df_success = filter(row -> row.status == "success", df)
println("Total successful runs: $(nrow(df_success))")

# Calculate total_elements for plotting
df_success.total_elements = df_success.num_tasks .* df_success.elements_per_task

# ------------------------------------------
# Plot 1: Single Task Performance (grouped bar plot)
# ------------------------------------------
println("\nGenerating Plot 1: Single Task Performance...")

single_task = filter(row -> row.benchmark_type == "single_task", df_success)

fig1 = Figure(; size=(800, 600))
ax1 = Axis(
    fig1[1, 1];
    xlabel="Elements per Task",
    ylabel="Time (ms)",
    title="Single Task Sorting Performance",
)

# Prepare data for grouped bar plot
elements = single_task.elements_per_task
n_groups = length(elements)

# Group the data by method
bitonic_times = single_task.bitonic_sort_time_ms
merge_times = single_task.merge_sort_time_ms
cpu_times = single_task.cpu_sort_time_ms

# Bar positions: each group centered at integer positions
bar_width = 0.25
gap = 0.05

for (i, elem_size) in enumerate(elements)
    subset = filter(row -> row.elements_per_task == elem_size, single_task)

    center = i
    # Three bars per group: CPU, GPU BitonicSort, GPU MergeSort
    cpu_pos = center - bar_width - gap/2
    bitonic_pos = center
    merge_pos = center + bar_width + gap/2

    cpu_time = first(subset.cpu_sort_time_ms)
    bitonic_time = first(subset.bitonic_sort_time_ms)
    merge_time = first(subset.merge_sort_time_ms)

    barplot!(ax1, [cpu_pos], [cpu_time];
             label=(i == 1) ? "CPU" : nothing, width=bar_width,
             color=:steelblue3, strokewidth=0)

    barplot!(ax1, [bitonic_pos], [bitonic_time];
             label=(i == 1) ? "GPU BitonicSort" : nothing, width=bar_width,
             color=:sandybrown, strokewidth=0)

    barplot!(ax1, [merge_pos], [merge_time];
             label=(i == 1) ? "GPU MergeSort" : nothing, width=bar_width,
             color=:darkgray, strokewidth=0)
end

# Set x-ticks to show element sizes
ax1.xticks = (1:n_groups, string.(elements))
xlims!(ax1, 0.5, n_groups + 0.5)

axislegend(ax1; position=:lt)

save(joinpath(OUTPUT_DIR, "01_single_task_performance.png"), fig1)
println("  Saved: ", joinpath(OUTPUT_DIR, "01_single_task_performance.png"))

# ------------------------------------------
# Plot 2: Multi-Task Method Comparison (fused from old plots 2 & 3)
# ------------------------------------------
println("\nGenerating Plot 2: Multi-Task Method Comparison...")

multi_task = filter(row -> row.benchmark_type == "multi_task", df_success)

# Filter to only include specific element sizes
sizes_to_keep = [10, 100, 500, 1000, 2000, 4000]
multi_task = filter(row -> row.elements_per_task in sizes_to_keep, multi_task)
unique_sizes = sort(unique(multi_task.elements_per_task))

# Markers for different element sizes
size_markers = Dict(
    10 => :circle,
    100 => :diamond,
    500 => :rect,
    1000 => :utriangle,
    2000 => :dtriangle,
    4000 => :pentagon
)

# Line styles for different element sizes
size_linestyles = Dict(
    10 => :dashdotdot,
    100 => :dash,
    500 => :dot,
    1000 => :dashdot,
    2000 => :dash,
    4000 => :solid
)

fig2 = Figure(; size=(800, 600))
ax2 = Axis(
    fig2[1, 1];
    xscale=log10,
    yscale=log10,
    xlabel="Number of Tasks",
    ylabel="Time (ms)",
    title="Multi-Task Sorting Method Comparison",
)

# First plot lines for each element size
for size in unique_sizes
    subset = filter(row -> row.elements_per_task == size, multi_task)
    linestyle = get(size_linestyles, size, :solid)

    # CPU (now first, with steelblue color)
    cpu_subset = filter(row -> !isnan(row.cpu_sort_time_ms), subset)
    if nrow(cpu_subset) > 0
        label = (size == unique_sizes[1]) ? "Batch CPU" : nothing
        lines!(ax2, cpu_subset.num_tasks, cpu_subset.cpu_sort_time_ms;
                label=label, linestyle=linestyle, color=:steelblue3, linewidth=2)
    end

    # BitonicSort (now second, with sandybrown color)
    bitonic_subset = filter(row -> !isnan(row.bitonic_sort_time_ms), subset)
    if nrow(bitonic_subset) > 0
        label = (size == unique_sizes[1]) ? "Batch BitonicSort" : nothing
        lines!(ax2, bitonic_subset.num_tasks, bitonic_subset.bitonic_sort_time_ms;
                label=label, linestyle=linestyle, color=:sandybrown, linewidth=2)
    end

    # MergeSort (now third, with darkgray color)
    merge_subset = filter(row -> !isnan(row.merge_sort_time_ms), subset)
    if nrow(merge_subset) > 0
        label = (size == unique_sizes[1]) ? "Sequential MergeSort (buffer reuse)" : nothing
        lines!(ax2, merge_subset.num_tasks, merge_subset.merge_sort_time_ms;
                label=label, linestyle=linestyle, color=:darkgray, linewidth=2)
    end
end

# Then overlay scatter points on top
for size in unique_sizes
    subset = filter(row -> row.elements_per_task == size, multi_task)
    marker = get(size_markers, size, :circle)

    # CPU (now first, with steelblue color)
    cpu_subset = filter(row -> !isnan(row.cpu_sort_time_ms), subset)
    if nrow(cpu_subset) > 0
        scatter!(ax2, cpu_subset.num_tasks, cpu_subset.cpu_sort_time_ms;
                 marker=marker, markersize=14, color=:steelblue3,
                 strokewidth=1, strokecolor=:white)
    end

    # BitonicSort (now second, with sandybrown color)
    bitonic_subset = filter(row -> !isnan(row.bitonic_sort_time_ms), subset)
    if nrow(bitonic_subset) > 0
        scatter!(ax2, bitonic_subset.num_tasks, bitonic_subset.bitonic_sort_time_ms;
                 marker=marker, markersize=14, color=:sandybrown,
                 strokewidth=1, strokecolor=:white)
    end

    # MergeSort (now third, with darkgray color)
    merge_subset = filter(row -> !isnan(row.merge_sort_time_ms), subset)
    if nrow(merge_subset) > 0
        scatter!(ax2, merge_subset.num_tasks, merge_subset.merge_sort_time_ms;
                 marker=marker, markersize=14, color=:darkgray,
                 strokewidth=1, strokecolor=:white)
    end
end

# Method legend (top left)
axislegend(ax2; position=:lt)

# Element size legend (bottom right, inside plot area)
# Combine line and marker elements for each size
size_legend_elements = [
    [
        LineElement(
            linestyle=get(size_linestyles, size, :solid),
            linewidth=2,
            linecolor=:black
        ),
        MarkerElement(
            marker=get(size_markers, size, :circle),
            markersize=12,
            color=:black,
            strokecolor=:white,
            strokewidth=1
        )
    ]
    for size in sort(unique_sizes, rev=true)
]
size_legend_labels = ["$size" for size in sort(unique_sizes, rev=true)]

fig2[1, 1, Right()] = Legend(
    fig2,
    size_legend_elements,
    size_legend_labels,
    "Elements/Task",
    margin=(5, 5, 5, 5),
    labelsize=10,
    titlesize=11,
    framevisible=true
)

save(joinpath(OUTPUT_DIR, "02_multitask_comparison.png"), fig2)
println("  Saved: ", joinpath(OUTPUT_DIR, "02_multitask_comparison.png"))

