"""Define sql queries to calculate similarity metrics between two columns in a table, and store the results in a new table."""


def cts_cts_query(col1: str, col2: str) -> str:
    """Create a SQL query to calculate similarity metrics between two columns in a table.

    This function generates a SQL query that calculates the following similarity metrics between two columns in a table:
    - Euclidean distance
    - Manhattan distance
    - Cosine similarity
    - Pearson correlation

    Parameters
    ----------
    col1, col2 : str
        The names of the columns to compare.

    Returns
    -------
    str
        The SQL query to calculate the similarity metrics. This should be
        passed to the duckdb.write() method to execute the query.
    """
    return f"""
        create schema if not exists cts_cts;
        create or replace table cts_cts.{col1}__{col2} as (
            with

            c1c2 as (
                select {col1}, {col2} from cluster.X
            ),

            std as (
                select
                    ((({col1} - avg({col1}) over ())) / (stddev({col1}) over ())) as {col1},
                    ((({col2} - avg({col2}) over ())) / (stddev({col2}) over ())) as {col2}

                from c1c2
            ),

            euclidean_distance as (
                select
                    *,
                    sqrt(
                        sum(
                            ({col1} - {col2}) ** 2
                        ) over ()
                    ) as euclidean_distance
                from std
            ),

            manhattan_distance as (
                select
                    *,
                    sum(
                        abs({col1} - {col2})
                    ) over () as manhattan_distance
                from euclidean_distance
            ),

            add_dot_product_and_norm as (
                select
                    {col1},
                    {col2},
                    euclidean_distance,
                    manhattan_distance,
                    sum({col1} * {col2}) over () as dot_product,
                    sqrt(sum({col1} ** 2) over ()) as norm1,
                    sqrt(sum({col2} ** 2) over ()) as norm2

                from manhattan_distance
            ),

            cosine_similarity as (
                select
                    *,
                    dot_product / (norm1 * norm2) as cosine_similarity
                    
                from add_dot_product_and_norm
            ),

            pearson_correlation as (
                select
                    *,
                    corr({col1}, {col2}) over () as pearson_correlation

                from cosine_similarity
            ),

            add_intersection_and_union as (
                select
                    *,
                    sum(try_cast({col1} = {col2} as double)) over () as _intersection,
                    try_cast((count(*) over ()) as double) as _union


                from pearson_correlation
            ),

        
            jaccard_similarity as (
                select
                    *,
                    case
                        when _union = 0 then 0
                        else _intersection / _union
                    end as jaccard_similarity

                from add_intersection_and_union
            )

            from jaccard_similarity
        );
        """  # noqa: S608


def binary_binary_query(col1: str, col2: str) -> str:
    """Create a SQL query to calculate similarity metrics between two columns in a table.

    This function generates a SQL query that calculates the following similarity metrics between two columns in a table:
    - Jaccard similarity
    - Hamming distance

    Parameters
    ----------
    col1, col2 : str
        The names of the columns to compare.

    Returns
    -------
    str
        The SQL query to calculate the similarity metrics. This should be
        passed to the duckdb.write() method to execute the query.
    """
    return f"""
        create schema if not exists bin_bin;
        create or replace table bin_bin.{col1}__{col2} as (
            with

            c1c2 as (
                select {col1}, {col2} from cluster.X
            ),

            add_intersection_and_union as (
                select
                    *,
                    sum(try_cast({col1} = {col2} as double)) over () as _intersection,
                    try_cast((count(*) over ()) as double) as _union


                from c1c2
            ),
        
            jaccard_similarity as (
                select
                    *,
                    case
                        when _union = 0 then 0
                        else _intersection / _union
                    end as jaccard_similarity

                from add_intersection_and_union
            ),

            hamming_distance as (
                select
                    *,
                    sum(try_cast({col1} != {col2} as double)) over () as hamming_distance

                from jaccard_similarity
            )

            from hamming_distance
        );
        """  # noqa: S608


def binary_categorical_query(col1: str, col2: str) -> str:
    """Create a SQL query to calculate similarity metrics between two columns in a table.

    This function generates a SQL query that calculates the following similarity metrics between two columns in a table:
    - Dice similarity
    - Cramér's V

    Parameters
    ----------
    col1, col2 : str
        The names of the columns to compare.

    Returns
    -------
    str
        The SQL query to calculate the similarity metrics. This should be
        passed to the duckdb.write() method to execute the query.
    """
    return f"""
        create schema if not exists bin_cat;
        create or replace table bin_cat.{col1}__{col2} as (
            with

            c1c2 as (
                select {col1}, {col2} from cluster.X
            ),

            add_intersection_and_union as (
                select
                    *,
                    sum(try_cast(({col1} = {col2}) as double)) over () as _intersection,
                    try_cast((count(*) over ()) as double) as _union


                from c1c2
            ),

            dice_similarity as (
                select
                    *,
                    (2 * _intersection) / (
                        (count(distinct {col1}) over ()) + (count(distinct {col2}) over ())
                    )  as dice_similarity

                from add_intersection_and_union
            ),

            count_col1 as (
                select
                    {col1},
                    count(*) as count_col1
                from cluster.X
                group by {col1}
            ),

            count_col2 as (
                select
                    {col2},
                    count(*) as count_col2
                from cluster.X
                group by {col2}
            ),

            count_col1_col2 as (
                select
                    {col1},
                    {col2},
                    count(*) as count_col1_col2
                from cluster.X
                group by {col1}, {col2}
            ),

            cramers_v_inputs as (
                select
                    dice.*,
                    count_col1.count_col1,
                    count_col2.count_col2,
                    count_col1_col2.count_col1_col2

                from
                    dice_similarity as dice
                    left join count_col1 
                        on dice.{col1} = count_col1.{col1}
                    left join count_col2
                        on dice.{col2} = count_col2.{col2}
                    left join count_col1_col2
                        on dice.{col1} = count_col1_col2.{col1}
                        and dice.{col2} = count_col1_col2.{col2}
            ),

            chi_squared as (
                with

                inputs as (
                    select
                        count_col1 as n1,
                        count_col2 as n2,
                        count_col1_col2 as n12,
                        count_col1 + count_col2 as n

                    from cramers_v_inputs
                ),

                add_numerator_and_denominator as (
                    select
                        (n12 - (n1 * n2 / n)) ** 2 as numerator,
                        (n1 * n2 / n) as denominator

                    from inputs
                )

                select sum(numerator / denominator) as chi_squared
                from add_numerator_and_denominator
            ),

            min_rows_and_columns as (
                select
                    case
                        when count_col1 < count_col2 then count_col1 - 1
                        else count_col2 - 1
                    end as min_,

                from cramers_v_inputs
            ),
        
            cramers_v as (
                select
                    chi_squared.chi_squared as chi_squared,
                    min_rows_and_columns.min_ as min_,
                    count(cvi.*) over () as n,
                    sqrt(chi_squared / (n * min_)) as cramers_v

                from cramers_v_inputs as cvi
                left join chi_squared on true
                left join min_rows_and_columns on true
            ),

            rejoin as (
                select
                    dice.*,
                    cramers_v.cramers_v

                from
                    dice_similarity as dice
                    left join cramers_v
                        on true
            )
                    
            from add_intersection_and_union
        );
        """  # noqa: S608
